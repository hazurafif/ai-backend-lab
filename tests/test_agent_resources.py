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
        assert created["path"] == "/skills/web-research/SKILL.md"

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


async def test_skill_file_visible_to_backend(persistence):
    """Skills written via the API land in the shared backend (/skills/ route)."""
    from app.schema.agent_schema import SkillIn
    from app.services.resources import create_skill

    backend = build_backend(store=persistence.store)
    await create_skill(
        persistence.store, SkillIn(name="api-skill", description="d", content="body")
    )
    files = await backend.adownload_files(["/skills/api-skill/SKILL.md"])
    assert files and files[0] is not None
    text = files[0].content.decode()
    assert "name: api-skill" in text and "body" in text

    # the agent builds and runs fine with the shared backend + skills source
    agent = build_agent(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=backend,
        model=scripted_model(),
        system_prompt="test",
    )
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, "tester", message="hi"):
        events.append(parse_sse_chunk(chunk))
    assert any(e == "done" for e, _ in events), events


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


async def test_skill_files_visible_to_backend(persistence):
    """Bundled files land in the shared /skills/ backend the agent reads."""
    from app.schema.agent_schema import SkillFileIn, SkillIn
    from app.services.resources import create_skill

    backend = build_backend(store=persistence.store)
    await create_skill(
        persistence.store,
        SkillIn(
            name="multi-file",
            description="d",
            content="body",
            files=[SkillFileIn(path="scripts/run.py", content="print('hi')")],
        ),
    )
    # SKILL.md + bundled file are both readable through the backend
    files = await backend.adownload_files(
        ["/skills/multi-file/SKILL.md", "/skills/multi-file/scripts/run.py"]
    )
    assert len(files) == 2 and all(f is not None for f in files)
    assert b"print('hi')" in files[1].content
    # directory listing exposes the skill-creator structure
    listing = await backend.als("/skills/multi-file")
    paths = [e["path"] for e in listing.entries]
    assert "/skills/multi-file/SKILL.md" in paths, paths
    assert "/skills/multi-file/scripts/" in paths, paths


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
            assert r.json() == {"connected": [], "tools": 0}
