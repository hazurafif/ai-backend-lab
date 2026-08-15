"""Offline tests for the agent resources CRUD API (skills, MCP tool servers).

Everything is persisted in the LangGraph store (in-memory here; Postgres in
production via the same BaseStore API). Verifies:

  - skill CRUD via HTTP, including the SKILL.md file shape the agent's
    SkillsMiddleware reads (store key /<name>/SKILL.md, {"content": ...})
  - skills survive across agent builds and are picked up from the shared
    backend (`/skills/` route of the CompositeBackend)
  - MCP tool server CRUD + store-first config loading (env/file fallback)
  - /agent/tools/reconnect rebuilds the agent from store configs
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from app.core import config, database
from app.core.constants import GLOBAL_SKILLS_NS
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent, build_backend
from app.services.chat import agent_stream

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class Scripted(BaseChatModel):
    """Returns a scripted sequence of AIMessages, clamping at the last."""

    responses: list[AIMessage] = Field(default_factory=list)
    tools: Sequence[dict | type] = ()
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
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
        self.tools = tools
        return self


def scripted_model() -> Scripted:
    return Scripted(responses=[AIMessage(content="Final answer from the agent.")])


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


async def client_for(app) -> httpx.AsyncClient:
    # Agent resource routes are admin-only and validated against the users
    # store, so seed the tester as an admin inside the lifespan context
    # (persistence.start() re-initializes the store on entry).
    await database.persistence.users.create_user(
        username="tester", hashed_password="x", role="admin"
    )
    token = create_access_token(data={"sub": "tester"})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


async def test_skill_crud(persistence):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await client_for(app) as http:
        # create
        r = await http.post(
            "/agent/skills",
            json={
                "name": "web-research",
                "description": "Research a topic",
                "content": "## Steps\n1. Search\n2. Synthesize",
            },
        )
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["path"] == "/skills/@tester/web-research/SKILL.md"

        # duplicate -> 409
        r = await http.post(
            "/agent/skills", json={"name": "web-research", "description": "x", "content": "y"}
        )
        assert r.status_code == 409, r.text

        # get + list
        r = await http.get("/agent/skills/web-research")
        assert r.status_code == 200
        assert "name: web-research" in r.json()["content"]
        r = await http.get("/agent/skills")
        assert [s["name"] for s in r.json()] == ["web-research"]

        # update
        r = await http.put(
            "/agent/skills/web-research",
            json={"name": "web-research", "description": "New desc", "content": "New body"},
        )
        assert r.status_code == 200
        assert "New body" in r.json()["content"]

        # invalid name rejected by schema
        r = await http.post(
            "/agent/skills", json={"name": "Bad Name!", "description": "x", "content": "y"}
        )
        assert r.status_code == 422, r.text

        # delete + 404 afterwards
        r = await http.delete("/agent/skills/web-research")
        assert r.status_code == 204
        r = await http.get("/agent/skills/web-research")
        assert r.status_code == 404

    # store shape: the agent's SkillsMiddleware reads {"content": ...} files
    item = await persistence.store.aget(GLOBAL_SKILLS_NS, "/web-research/SKILL.md")
    assert item is None  # deleted


async def test_skill_file_visible_to_backend(persistence, tmp_path, monkeypatch):
    """The user's own skills land in their workspace; the global pool does not."""
    from app.core.constants import user_skills_ns
    from app.schema.agent_schema import SkillIn
    from app.services.resources import create_skill
    from app.services.workspace import materialize_skills

    monkeypatch.setattr(config.settings, "workspace_root", str(tmp_path / "workspaces"))
    # A global (admin) skill must NOT leak into the user's workspace...
    await create_skill(
        persistence.store, SkillIn(name="global-skill", description="d", content="g")
    )
    # ...while the user's own skill does.
    await create_skill(
        persistence.store,
        SkillIn(name="api-skill", description="d", content="body"),
        user_skills_ns("tester"),
    )
    await materialize_skills(persistence.store, "tester")
    skill_dir = tmp_path / "workspaces" / "tester" / "skills"
    assert (skill_dir / "api-skill" / "SKILL.md").exists()
    assert "body" in (skill_dir / "api-skill" / "SKILL.md").read_text()
    assert not (skill_dir / "global-skill").exists(), "global skills must be isolated"

    # the agent builds and runs fine with the workspace backend + skills source
    agent = build_agent(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
        model=scripted_model(),
        system_prompt="test",
    )
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, "tester", message="hi"):
        events.append(parse_sse_chunk(chunk))
    assert any(e == "done" for e, _ in events), events


async def test_backend_is_single_per_user_workspace(persistence, monkeypatch, tmp_path):
    """The backend is one UserShellBackend: everything lands in the user's dir.

    No virtual mounts: file-tool writes under any path (and execute commands)
    resolve to WORKSPACE_ROOT/<user>/, isolated per user.
    """
    import app.services.settings as runtime_settings
    from app.services.agent import UserShellBackend

    monkeypatch.setattr(runtime_settings, "execute_enabled", lambda: True)
    monkeypatch.setattr(runtime_settings, "execute_inherit_env", lambda: False)
    monkeypatch.setattr(config.settings, "workspace_root", str(tmp_path / "workspaces"))

    backend = build_backend(store=persistence.store)
    assert isinstance(backend.default, UserShellBackend)
    assert backend.routes == {}

    # Outside a graph run the runtime user falls back to "anonymous".
    result = await backend.awrite("/notes.txt", "hi")
    assert result.error is None, result.error
    assert (tmp_path / "workspaces" / "anonymous" / "notes.txt").exists()

    # Execute is refused when the opt-in is off (same backend, gated tool).
    monkeypatch.setattr(runtime_settings, "execute_enabled", lambda: False)
    backend = build_backend(store=persistence.store)
    resp = backend.default.execute("echo hi")  # type: ignore[attr-defined]
    assert resp.exit_code == 1
    assert "Execution not available" in resp.output


async def test_workspace_isolates_users_and_is_shell_visible(persistence, tmp_path, monkeypatch):
    """End-to-end: write_file + execute agree on /workspaces/<user>/ files.

    The file written through the file tool lands in the runtime user's dir
    and the execute tool's shell (cwd = that dir) can see it — the store
    routes' "virtual mount" split does not apply to the workspace.
    """
    import app.services.settings as runtime_settings
    from app.services.agent import build_agent
    from app.services.chat import agent_stream as chat_agent_stream

    monkeypatch.setattr(runtime_settings, "execute_enabled", lambda: True)
    monkeypatch.setattr(runtime_settings, "execute_inherit_env", lambda: False)
    monkeypatch.setattr(config.settings, "workspace_root", str(tmp_path / "workspaces"))

    class ScriptedWriteThenRun(BaseChatModel):
        """write_file /script.py, then run `test -f` via execute."""

        responses: list[AIMessage] = Field(
            default_factory=lambda: [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="write_file",
                            args={
                                "file_path": "/script.py",
                                "content": "print('hello from workspace')",
                            },
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-2",
                            name="execute",
                            args={"command": "test -f script.py && echo FOUND_IT"},
                        )
                    ],
                ),
                AIMessage(content="done"),
            ]
        )

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def _generate(
            self,
            messages: LanguageModelInput,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

        def bind_tools(
            self,
            tools: Sequence[dict | type | BaseChatModel],
            *,
            tool_choice: str | None = None,
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, AIMessage]:
            return self

    agent = build_agent(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        model=ScriptedWriteThenRun(),
        system_prompt="test",
    )
    events: list[tuple[str, dict]] = []
    async for chunk in chat_agent_stream(agent, "alice", message="go"):
        ev, _, rest = chunk.partition("\n")
        events.append((ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())))

    # The file landed in ALICE's dir, not a shared one.
    script = tmp_path / "workspaces" / "alice" / "script.py"
    assert script.exists()
    assert "hello from workspace" in script.read_text()
    # execute ran with cwd = her dir and saw the file via the shell.
    tool_outputs = [d for e, d in events if e == "tool_end"]
    assert any("FOUND_IT" in str(d.get("output", {})) for d in tool_outputs)
    # The run auto-committed the workspace to its git repo (best-effort,
    # may finish after the stream closes — poll briefly).
    import subprocess

    def _git_log() -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path / "workspaces"), "log", "--oneline"],
            capture_output=True,
            text=True,
        ).stdout

    git_log = None
    for _ in range(100):
        git_log = await asyncio.to_thread(_git_log)
        if git_log.strip():
            break
        await asyncio.sleep(0.05)
    assert git_log is not None and git_log.strip(), git_log
    assert "run " in git_log


async def test_skill_bundled_files_crud(persistence):
    """Skills support the skill-creator layout: scripts/, references/, assets/..."""
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await client_for(app) as http:
        # create with bundled files (sorted by path in the response)
        r = await http.post(
            "/agent/skills",
            json={
                "name": "pdf-tool",
                "description": "PDF utilities",
                "content": "## Steps",
                "files": [
                    {"path": "scripts/extract.py", "content": "print('extract')"},
                    {"path": "references/spec.md", "content": "# Spec"},
                    {"path": "assets/logo.svg", "content": "<svg/>"},
                ],
            },
        )
        assert r.status_code == 201, r.text
        assert [f["path"] for f in r.json()["files"]] == [
            "assets/logo.svg",
            "references/spec.md",
            "scripts/extract.py",
        ]

        # get returns contents; list shows the file tree
        r = await http.get("/agent/skills/pdf-tool")
        files = {f["path"]: f["content"] for f in r.json()["files"]}
        assert files["scripts/extract.py"] == "print('extract')"
        assert files["references/spec.md"] == "# Spec"
        r = await http.get("/agent/skills")
        assert [s["name"] for s in r.json()] == ["pdf-tool"]
        assert r.json()[0]["files"][0]["path"] == "assets/logo.svg"

        # update replaces listed files, keeps unlisted ones
        r = await http.put(
            "/agent/skills/pdf-tool",
            json={
                "name": "pdf-tool",
                "description": "d2",
                "content": "New body",
                "files": [{"path": "scripts/extract.py", "content": "print('v2')"}],
            },
        )
        assert r.status_code == 200, r.text
        files = {f["path"]: f["content"] for f in r.json()["files"]}
        assert files["scripts/extract.py"] == "print('v2')"
        assert "references/spec.md" in files, "unlisted files must be kept"

        # delete a single file
        r = await http.delete("/agent/skills/pdf-tool/files/references/spec.md")
        assert r.status_code == 204, r.text
        r = await http.get("/agent/skills/pdf-tool")
        assert "references/spec.md" not in [f["path"] for f in r.json()["files"]]
        r = await http.delete("/agent/skills/pdf-tool/files/references/spec.md")
        assert r.status_code == 404

        # deleting SKILL.md via the file endpoint is rejected
        r = await http.delete("/agent/skills/pdf-tool/files/SKILL.md")
        assert r.status_code == 422, r.text

        # path traversal / malformed paths rejected
        for bad in ("../evil", "/etc/passwd", "scripts//x.py", "a b.py", ".."):
            r = await http.post(
                "/agent/skills",
                json={
                    "name": "bad-skill",
                    "description": "d",
                    "content": "c",
                    "files": [{"path": bad, "content": "x"}],
                },
            )
            assert r.status_code == 422, bad

        # delete removes SKILL.md AND all bundled files
        r = await http.delete("/agent/skills/pdf-tool")
        assert r.status_code == 204
    for key in ("/pdf-tool/SKILL.md", "/pdf-tool/scripts/extract.py", "/pdf-tool/assets/logo.svg"):
        assert await persistence.store.aget(GLOBAL_SKILLS_NS, key) is None


async def test_skill_files_visible_to_backend(persistence, tmp_path, monkeypatch):
    """Bundled files land in the workspace the agent reads (materialized)."""
    from app.core.constants import user_skills_ns
    from app.schema.agent_schema import SkillFileIn, SkillIn
    from app.services.resources import create_skill
    from app.services.workspace import materialize_skills

    monkeypatch.setattr(config.settings, "workspace_root", str(tmp_path / "workspaces"))
    await create_skill(
        persistence.store,
        SkillIn(
            name="multi-file",
            description="d",
            content="body",
            files=[SkillFileIn(path="scripts/run.py", content="print('hi')")],
        ),
        user_skills_ns("tester"),
    )
    await materialize_skills(persistence.store, "tester")
    # SKILL.md + bundled file are both real files in the workspace.
    skill_dir = tmp_path / "workspaces" / "tester" / "skills" / "multi-file"
    assert (skill_dir / "SKILL.md").exists()
    assert b"print('hi')" in (skill_dir / "scripts" / "run.py").read_bytes()
    # A graph run (runtime user = tester) resolves /skills/ into that dir.
    agent = build_agent(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
        model=scripted_model(),
        system_prompt="test",
    )
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, "tester", message="hi"):
        events.append(parse_sse_chunk(chunk))
    assert any(e == "done" for e, _ in events), events


# ---------------------------------------------------------------------------
# MCP tool servers
# ---------------------------------------------------------------------------


async def test_tool_server_crud_and_store_first_config(persistence):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await client_for(app) as http:
        # create a stdio server and a disabled http server
        r = await http.post(
            "/agent/tools",
            json={
                "name": "cli-tool",
                "transport": "stdio",
                "command": "gofastmcp-tool",
                "args": ["serve"],
            },
        )
        assert r.status_code == 201, r.text
        r = await http.post(
            "/agent/tools",
            json={
                "name": "weather",
                "transport": "streamable_http",
                "url": "http://localhost:8090/mcp",
                "enabled": False,
            },
        )
        assert r.status_code == 201, r.text

        # get/list
        r = await http.get("/agent/tools/cli-tool")
        assert r.json()["command"] == "gofastmcp-tool"
        r = await http.get("/agent/tools")
        assert {s["name"] for s in r.json()} == {"cli-tool", "weather"}

        # update + delete
        r = await http.put(
            "/agent/tools/cli-tool",
            json={"name": "cli-tool", "transport": "stdio", "command": "other-tool"},
        )
        assert r.status_code == 200 and r.json()["command"] == "other-tool"
        r = await http.delete("/agent/tools/weather")
        assert r.status_code == 204

    # config loading: store wins over env/file; disabled entries skipped
    from app.services.resources import load_tool_server_configs

    cfg = await load_tool_server_configs(persistence.store)
    assert list(cfg) == ["cli-tool"], cfg
    assert cfg["cli-tool"]["command"] == "other-tool"


async def test_reconnect_rebuilds_agent(persistence, monkeypatch):
    monkeypatch.setattr(config.settings, "model", scripted_model())
    app = create_app()
    async with app.router.lifespan_context(app):
        assert hasattr(app.state, "agent")
        async with await client_for(app) as http:
            r = await http.post("/agent/tools/reconnect")
            assert r.status_code == 200, r.text
            assert r.json() == {"connected": [], "tools": 0, "failed": {}}
