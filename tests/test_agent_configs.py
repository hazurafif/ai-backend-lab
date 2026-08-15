"""Offline tests for agent configs (customizable agent profiles).

Covers:

  - /agents CRUD via HTTP: user agents, admin-only global scope, reserved
    'default' name, name collisions, unknown skill/tool validation
  - chat routing: POST /chat with `agent` uses that profile (system prompt),
    thread metadata records the agent, unknown agent -> 404
  - per-agent skills: snapshot copy into the agent's namespace + isolation
    (an agent with skills=[] does NOT see global skills)
  - per-agent tool selection by MCP server name (+ web_search pseudo-tool)
  - registry caching: same fingerprint -> same graph, invalidate -> rebuild
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from pydantic import Field

from app.core import config, database
from app.core.constants import agent_skills_ns
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import AgentRegistry, build_backend

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class RecordingModel(BaseChatModel):
    """Scripted model that records system prompts and bound tool names."""

    system_prompts: list[str] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    responses: list[AIMessage] = Field(
        default_factory=lambda: [AIMessage(content="Final answer from the agent.")]
    )
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "recording"

    @staticmethod
    def _text(content: Any) -> str:
        """System message content can be a plain string or content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content)

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        for m in messages:
            if getattr(m, "type", None) == "system":
                self.system_prompts.append(self._text(m.content))
        i = min(self._idx, len(self.responses) - 1)
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseChatModel],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = [
            t.name if hasattr(t, "name") else (t.get("name") if isinstance(t, dict) else str(t))
            for t in tools
        ]
        return self


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


async def client_for(app, username: str, role: str = "user") -> httpx.AsyncClient:
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def make_app(
    persistence, *, model: RecordingModel, mcp_tools=None, tools_by_server=None, extra_tools=None
):
    """App whose registry builds graphs with the recording model."""
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
        mcp_tools=mcp_tools or [],
        extra_tools=extra_tools or [],
        tools_by_server=tools_by_server or {},
        model_factory=lambda m, t: model,
    )
    return create_app(agent_registry=registry)


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


def agent_payload(name: str, **overrides) -> dict:
    payload = {
        "name": name,
        "model": "openai:gpt-4o-mini",
        "description": "test agent",
        "system_prompt": f"You are the {name} agent.",
        "skills": None,
        "tools": None,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# /agents CRUD
# ---------------------------------------------------------------------------


async def test_agent_config_crud(persistence):
    app = make_app(persistence, model=RecordingModel())
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        # create
        r = await client.post("/agents", json=agent_payload("research"))
        assert r.status_code == 201, r.text
        assert r.json()["scope"] == "user"
        assert r.json()["owner"] == "alice"

        # duplicate -> 409
        r = await client.post("/agents", json=agent_payload("research"))
        assert r.status_code == 409, r.text

        # reserved name -> 400
        r = await client.post("/agents", json=agent_payload("default"))
        assert r.status_code == 400, r.text

        # list: builtin default first, then alice's
        r = await client.get("/agents")
        assert r.status_code == 200, r.text
        agents = r.json()
        assert [a["name"] for a in agents] == ["default", "research"]
        assert agents[0]["builtin"] is True

        # get
        r = await client.get("/agents/research")
        assert r.status_code == 200 and r.json()["system_prompt"] == "You are the research agent."

        # update
        r = await client.put(
            "/agents/research", json=agent_payload("research", system_prompt="v2 prompt")
        )
        assert r.status_code == 200 and r.json()["system_prompt"] == "v2 prompt"

        # delete
        r = await client.delete("/agents/research")
        assert r.status_code == 204
        r = await client.get("/agents/research")
        assert r.status_code == 404
        r = await client.delete("/agents/research")
        assert r.status_code == 404


async def test_agent_config_validation_and_scopes(persistence):
    app = make_app(persistence, model=RecordingModel())
    async with app.router.lifespan_context(app):
        # unknown skill -> 400 (BadRequest)
        async with await client_for(app, "alice") as client:
            r = await client.post("/agents", json=agent_payload("x", skills=["does-not-exist"]))
            assert r.status_code == 400, r.text
            # unknown tool server -> 400
            r = await client.post("/agents", json=agent_payload("x", tools=["ghost-server"]))
            assert r.status_code == 400, r.text
            # non-admin cannot create global agents -> 403
            r = await client.post("/agents", json=agent_payload("x", scope="global"))
            assert r.status_code == 403, r.text

        # admin can
        async with await client_for(app, "boss", role="admin") as admin:
            r = await admin.post("/agents", json=agent_payload("shared", scope="global"))
            assert r.status_code == 201, r.text
            assert r.json()["owner"] == "global"

        # other users see the global agent in their list
        async with await client_for(app, "bob") as bob:
            r = await bob.get("/agents")
            assert [a["name"] for a in r.json()] == ["default", "shared"]
            # ...but cannot update/delete it
            assert (
                await bob.put("/agents/shared", json=agent_payload("shared", scope="global"))
            ).status_code == 403
            assert (await bob.delete("/agents/shared")).status_code == 403


async def test_agent_ownership(persistence):
    app = make_app(persistence, model=RecordingModel())
    async with app.router.lifespan_context(app):
        async with await client_for(app, "alice") as alice:
            assert (await alice.post("/agents", json=agent_payload("mine"))).status_code == 201
        async with await client_for(app, "bob") as bob:
            # bob cannot read/update/delete alice's agent
            assert (await bob.get("/agents/mine")).status_code == 404
            assert (await bob.put("/agents/mine", json=agent_payload("mine"))).status_code == 404
            assert (await bob.delete("/agents/mine")).status_code == 404
            # but bob can create his own agent with the same name
            assert (await bob.post("/agents", json=agent_payload("mine"))).status_code == 201
            # alice still owns hers
            assert (await bob.get("/agents/mine")).json()["owner"] == "bob"


# ---------------------------------------------------------------------------
# chat routing
# ---------------------------------------------------------------------------


async def test_chat_routes_to_named_agent(persistence):
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        await client.post("/agents", json=agent_payload("research"))
        r = await client.post("/chat", json={"message": "hello", "agent": "research"})
        assert r.status_code == 200, r.text
        events = [parse_sse_chunk(c) for c in r.text.split("\n\n") if c.strip()]
        assert any(ev == "done" for ev, _ in events)

        # the run used the agent's system prompt
        assert any("research agent" in p for p in model.system_prompts), model.system_prompts

        # thread metadata records the agent name
        r = await client.get("/threads")
        assert r.status_code == 200
        threads = r.json()
        assert threads and threads[0]["agent"] == "research"

        # unknown agent -> 404 before streaming
        r = await client.post("/chat", json={"message": "hi", "agent": "nope"})
        assert r.status_code == 404, r.text


async def test_default_agent_keeps_env_behavior(persistence):
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        r = await client.post("/chat", json={"message": "hi"})
        assert r.status_code == 200
        # default agent = env settings (system prompt from settings)
        assert any("AI assistant" in p for p in model.system_prompts), model.system_prompts


# ---------------------------------------------------------------------------
# per-agent skills (snapshot + isolation)
# ---------------------------------------------------------------------------


async def test_agent_skills_snapshot_and_isolation(persistence):
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with (
        app.router.lifespan_context(app),
        await client_for(app, "alice") as client,
    ):
        # alice creates her OWN skill (skills are fully per-user)
        r = await client.post(
            "/skills",
            json={
                "name": "sql-guru",
                "description": "SQL expertise",
                "content": "You are an SQL expert. Prefer indexes.",
            },
        )
        assert r.status_code == 201, r.text

        # agent with the skill attached
        r = await client.post("/agents", json=agent_payload("dba", skills=["sql-guru"]))
        assert r.status_code == 201, r.text

        # snapshot exists in the agent's namespace (not only global)
        ns = agent_skills_ns("alice", "dba")
        items = [it for it in await persistence.store.asearch(ns)]
        assert items, "agent skill namespace is empty"
        md = next(it for it in items if it.key == "/sql-guru/SKILL.md")
        assert "SQL expert" in md.value["content"]

        # agent without skills must NOT see the global skill
        r = await client.post("/agents", json=agent_payload("plain", skills=[]))
        assert r.status_code == 201, r.text

        model.system_prompts.clear()
        r = await client.post("/chat", json={"message": "hi", "agent": "dba"})
        assert r.status_code == 200
        dba_prompts = list(model.system_prompts)
        assert any("sql-guru" in p for p in dba_prompts), dba_prompts

        model.system_prompts.clear()
        r = await client.post("/chat", json={"message": "hi", "agent": "plain"})
        assert r.status_code == 200
        plain_prompts = list(model.system_prompts)
        assert all("sql-guru" not in p for p in plain_prompts), plain_prompts


# ---------------------------------------------------------------------------
# per-agent tool selection
# ---------------------------------------------------------------------------


def _fake_tool(name: str) -> StructuredTool:
    async def _run(x: str) -> str:
        return f"{name}:{x}"

    return StructuredTool.from_function(coroutine=_run, name=name, description=f"fake {name} tool")


async def test_agent_tool_selection(persistence):
    model = RecordingModel()
    mcp_tools = [_fake_tool("tool_a"), _fake_tool("tool_b")]
    search = _fake_tool("web_search")
    app = make_app(
        persistence,
        model=model,
        mcp_tools=mcp_tools,
        tools_by_server={"srv-a": ["tool_a"], "srv-b": ["tool_b"]},
        extra_tools=[search],
    )
    # endpoint-side tool validation reads the live mcp_servers config; seed it
    # inside the lifespan (connect() would otherwise overwrite it)
    from app.services.mcp import mcp_servers

    saved_config = mcp_servers._config
    try:
        async with app.router.lifespan_context(app):
            mcp_servers._config = {"srv-a": {}, "srv-b": {}}
            async with await client_for(app, "alice") as client:
                # default agent: all tools
                r = await client.post("/chat", json={"message": "hi"})
                assert r.status_code == 200
                assert set(model.bound_tools) >= {"tool_a", "tool_b", "web_search"}

                # selected server only
                r = await client.post("/agents", json=agent_payload("slim", tools=["srv-a"]))
                assert r.status_code == 201, r.text
                model.bound_tools.clear()
                r = await client.post("/chat", json={"message": "hi", "agent": "slim"})
                assert r.status_code == 200
                assert "tool_a" in model.bound_tools, model.bound_tools
                assert "tool_b" not in model.bound_tools, model.bound_tools

                # web_search pseudo-tool
                r = await client.post("/agents", json=agent_payload("webby", tools=["web_search"]))
                assert r.status_code == 201, r.text
                model.bound_tools.clear()
                r = await client.post("/chat", json={"message": "hi", "agent": "webby"})
                assert r.status_code == 200
                assert "web_search" in model.bound_tools, model.bound_tools
                assert "tool_a" not in model.bound_tools, model.bound_tools

                # no tools
                r = await client.post("/agents", json=agent_payload("bare", tools=[]))
                assert r.status_code == 201, r.text
                model.bound_tools.clear()
                r = await client.post("/chat", json={"message": "hi", "agent": "bare"})
                assert r.status_code == 200
                assert "tool_a" not in model.bound_tools, model.bound_tools
                assert "web_search" not in model.bound_tools, model.bound_tools
    finally:
        mcp_servers._config = saved_config


# ---------------------------------------------------------------------------
# registry caching
# ---------------------------------------------------------------------------


async def test_registry_cache_and_invalidate(persistence):
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with app.router.lifespan_context(app):
        registry = app.state.agents
        async with await client_for(app, "alice") as client:
            await client.post("/agents", json=agent_payload("research"))

            g1 = await registry.resolve("research", "alice")
            g2 = await registry.resolve("research", "alice")
            assert g1 is g2, "same fingerprint must reuse the cached graph"

            registry.invalidate()
            g3 = await registry.resolve("research", "alice")
            assert g3 is not g1, "invalidate must drop cached graphs"

            # unknown agent
            try:
                await registry.resolve("ghost", "alice")
                raise AssertionError("expected KeyError")
            except KeyError:
                pass

            # default agent resolves per user (system prompt is rendered
            # with the username), cached per user, shared across repeats
            d1 = await registry.resolve("default", "alice")
            d2 = await registry.resolve("default", "alice")
            d3 = await registry.resolve("default", "bob")
            assert d1 is d2, "same user must reuse the cached graph"
            assert d1 is not d3, "different users get their own rendered prompt"


# ---------------------------------------------------------------------------
# user-scoped skills (/skills) + agent references
# ---------------------------------------------------------------------------


def skill_payload(name: str, content: str = "Skill body.") -> dict:
    return {"name": name, "description": f"{name} description", "content": content}


async def test_user_skills_crud(persistence):
    app = make_app(persistence, model=RecordingModel())
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        # create
        r = await client.post("/skills", json=skill_payload("my-skill"))
        assert r.status_code == 201, r.text
        assert r.json()["path"].startswith("/skills/@alice/")

        # duplicate -> 409
        assert (await client.post("/skills", json=skill_payload("my-skill"))).status_code == 409

        # list/get/update
        r = await client.get("/skills")
        assert [s["name"] for s in r.json()] == ["my-skill"]
        assert (await client.get("/skills/my-skill")).status_code == 200
        r = await client.put("/skills/my-skill", json=skill_payload("my-skill", "updated body"))
        assert r.status_code == 200 and "updated body" in r.json()["content"]

        # bundled file + delete file
        r = await client.post(
            "/skills",
            json={
                **skill_payload("with-files"),
                "files": [{"path": "scripts/run.py", "content": "print(1)"}],
            },
        )
        assert r.status_code == 201
        assert (await client.delete("/skills/with-files/files/scripts/run.py")).status_code == 204

        # delete
        assert (await client.delete("/skills/my-skill")).status_code == 204
        assert (await client.get("/skills/my-skill")).status_code == 404


async def test_user_skills_ownership(persistence):
    app = make_app(persistence, model=RecordingModel())
    async with app.router.lifespan_context(app):
        async with await client_for(app, "alice") as alice:
            assert (await alice.post("/skills", json=skill_payload("private"))).status_code == 201
        async with await client_for(app, "bob") as bob:
            # bob cannot see/update/delete alice's skill
            assert (await bob.get("/skills/private")).status_code == 404
            assert (
                await bob.put("/skills/private", json=skill_payload("private"))
            ).status_code == 404
            assert (await bob.delete("/skills/private")).status_code == 404
            # same name is fine for bob
            assert (await bob.post("/skills", json=skill_payload("private"))).status_code == 201


async def test_agent_can_reference_user_skill(persistence):
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        await client.post("/skills", json=skill_payload("sql-guru", "You are an SQL expert."))
        r = await client.post("/agents", json=agent_payload("dba", skills=["sql-guru"]))
        assert r.status_code == 201, r.text

        # snapshot copied from the USER namespace into the agent's namespace
        ns = agent_skills_ns("alice", "dba")
        md = await persistence.store.aget(ns, "/sql-guru/SKILL.md")
        assert md is not None and "SQL expert" in md.value["content"]

        # ...another user (no own skill of that name) cannot reference it:
        # skills are fully per-user, there is no global fallback.
        async with await client_for(app, "bob") as bob:
            r = await bob.post("/agents", json=agent_payload("dba2", skills=["sql-guru"]))
            assert r.status_code == 400, r.text
            assert "Unknown skill" in r.json()["detail"]

        # run it: the skill reaches the system prompt
        model.system_prompts.clear()
        r = await client.post("/chat", json={"message": "hi", "agent": "dba"})
        assert r.status_code == 200
        assert any("sql-guru" in p for p in model.system_prompts), model.system_prompts


async def test_global_agent_cannot_reference_user_skill(persistence):
    app = make_app(persistence, model=RecordingModel())
    async with (
        app.router.lifespan_context(app),
        await client_for(app, "boss", role="admin") as admin,
    ):
        await admin.post("/skills", json=skill_payload("private-skill"))
        r = await admin.post(
            "/agents", json=agent_payload("shared", scope="global", skills=["private-skill"])
        )
        assert r.status_code == 400, r.text


async def test_agent_dry_run_test_endpoint(persistence):
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        await client.post("/agents", json=agent_payload("research"))
        r = await client.post("/agents/research/test")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok" and body["graph_built"] is True
        assert body["model"] == "openai:gpt-4o-mini"

        # builtin default is testable too
        r = await client.post("/agents/default/test")
        assert r.status_code == 200, r.text

        # unknown agent -> 404
        assert (await client.post("/agents/ghost/test")).status_code == 404


async def test_system_prompt_renders_username_per_user(persistence):
    """The {{username}} placeholder is replaced with the real user per run."""
    from app.services.agent_configs import load_spec

    alice = await load_spec(persistence.store, "default", "alice")
    assert alice.system_prompt is not None
    assert "{{username}}" not in alice.system_prompt
    assert ".workspace/alice" in alice.system_prompt

    budi = await load_spec(persistence.store, "default", "budi")
    assert ".workspace/budi" in budi.system_prompt
    assert ".workspace/alice" not in budi.system_prompt

    # The model actually receives the rendered prompt during a chat.
    model = RecordingModel()
    app = make_app(persistence, model=model)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        r = await client.post("/chat", json={"message": "hi"})
        assert r.status_code == 200, r.text
    assert any(".workspace/alice" in p for p in model.system_prompts), model.system_prompts
