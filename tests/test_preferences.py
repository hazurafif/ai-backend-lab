"""Offline tests for per-user preferences: store, endpoints, chat fallback.

The web search toggle is persisted per user (`user_preferences` table /
in-memory fallback) so the frontend can drop localStorage: PATCH
/users/me/preferences stores it, chat requests that omit `enable_search`
fall back to the stored value. Precedence: request field > stored
preference > SEARXNG_ENABLED config.

Covers:
  - store: get/set/clear semantics (memory mode)
  - endpoints: GET returns the server default when unset, PATCH sets and
    clears, GET reflects the stored value
  - chat: stored preference applies when the body omits the field; an
    explicit field still wins over the stored preference
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from app.core import config
from app.core.database import persistence
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.searxng import build_search_tool

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)

SEARXNG_JSON = {
    "results": [
        {
            "title": "FastAPI Release Notes",
            "url": "https://fastapi.tiangolo.com/release-notes/",
            "content": "Latest FastAPI releases and changelog.",
            "engine": "google",
        }
    ],
    "number_of_results": 42,
}


def fake_searxng_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=SEARXNG_JSON)


class Scripted(BaseChatModel):
    """One-shot scripted model that calls web_search once, then answers."""

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
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="call-1", name="web_search", args={"query": "fastapi release"})
                ],
            ),
            AIMessage(content="Here is what I found."),
        ]
        i = min(self._idx, 1)
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=responses[i])])

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseChatModel],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def build_search_agent(client: httpx.AsyncClient) -> Any:
    return build_agent(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        mcp_tools=[build_search_tool(client=client)],
        model=Scripted(),
        system_prompt="test",
    )


async def start_memory_persistence():
    config.settings.database_uri = None
    await persistence.start()
    return persistence


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


async def test_store_set_get_clear():
    persistence = await start_memory_persistence()
    try:
        store = persistence.preferences
        assert await store.get("alice", "enable_search") is None
        assert await store.get_all("alice") == {}

        await store.set("alice", "enable_search", True)
        assert await store.get("alice", "enable_search") is True
        assert await store.get("bob", "enable_search") is None  # owner-scoped

        await store.set("alice", "enable_search", False)  # overwrite
        assert await store.get("alice", "enable_search") is False

        await store.set("alice", "enable_search", None)  # clear
        assert await store.get("alice", "enable_search") is None
        assert await store.get_all("alice") == {}
    finally:
        await persistence.stop()


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


async def test_get_preferences_returns_server_default_when_unset(monkeypatch):
    persistence = await start_memory_persistence()
    try:
        app = create_app()
        token = create_access_token(data={"sub": "alice"})
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http,
        ):
            await persistence.users.create_user(username="alice", hashed_password="x")
            headers = {"Authorization": f"Bearer {token}"}

            r = await http.get("/users/me/preferences", headers=headers)
            assert r.status_code == 200, r.text
            assert r.json() == {"enable_search": config.settings.searxng_enabled}

            # explicit request-level defaults are not part of preferences
            r = await http.get("/users/me/preferences", headers={"Authorization": "Bearer bad"})
            assert r.status_code == 401
    finally:
        await persistence.stop()


async def test_patch_preferences_sets_and_clears(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_enabled", True)
    persistence = await start_memory_persistence()
    try:
        app = create_app()
        token = create_access_token(data={"sub": "alice"})
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http,
        ):
            await persistence.users.create_user(username="alice", hashed_password="x")
            headers = {"Authorization": f"Bearer {token}"}

            r = await http.patch(
                "/users/me/preferences", json={"enable_search": False}, headers=headers
            )
            assert r.status_code == 200, r.text
            assert r.json() == {"enable_search": False}
            assert await persistence.preferences.get("alice", "enable_search") is False

            # GET reflects the stored value
            r = await http.get("/users/me/preferences", headers=headers)
            assert r.json() == {"enable_search": False}

            # explicit null clears back to the server default
            r = await http.patch(
                "/users/me/preferences", json={"enable_search": None}, headers=headers
            )
            assert r.status_code == 200, r.text
            assert r.json() == {"enable_search": True}
            assert await persistence.preferences.get("alice", "enable_search") is None

            # empty body is a no-op (omitted key untouched)
            r = await http.patch("/users/me/preferences", json={}, headers=headers)
            assert r.status_code == 200, r.text
            assert r.json() == {"enable_search": True}
    finally:
        await persistence.stop()


# ---------------------------------------------------------------------------
# chat fallback
# ---------------------------------------------------------------------------


async def test_chat_uses_stored_preference_when_field_omitted(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    monkeypatch.setattr(config.settings, "searxng_enabled", True)
    persistence = await start_memory_persistence()
    try:
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake_searxng_handler))
        app = create_app(agent=build_search_agent(client))
        token = create_access_token(data={"sub": "alice"})
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http,
        ):
            await persistence.users.create_user(username="alice", hashed_password="x")
            headers = {"Authorization": f"Bearer {token}"}

            # stored OFF -> omitted field disables search (fresh agent: the
            # scripted model is a one-shot sequence per run)
            await persistence.preferences.set("alice", "enable_search", False)
            r = await http.post("/chat", json={"message": "search please"}, headers=headers)
            assert r.status_code == 200, r.text
            assert "Web search is disabled" in r.text

            # stored ON -> omitted field runs the (fake) search
            await persistence.preferences.set("alice", "enable_search", True)
            app.state.agent = build_search_agent(client)
            app.state.agents.set_static_default(app.state.agent)
            r = await http.post("/chat", json={"message": "search please"}, headers=headers)
            assert r.status_code == 200, r.text
            assert "FastAPI Release Notes" in r.text
    finally:
        await persistence.stop()


async def test_chat_request_field_overrides_stored_preference(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    monkeypatch.setattr(config.settings, "searxng_enabled", True)
    persistence = await start_memory_persistence()
    try:
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake_searxng_handler))
        app = create_app(agent=build_search_agent(client))
        token = create_access_token(data={"sub": "alice"})
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http,
        ):
            await persistence.users.create_user(username="alice", hashed_password="x")
            headers = {"Authorization": f"Bearer {token}"}

            # stored OFF but explicit true -> search runs
            await persistence.preferences.set("alice", "enable_search", False)
            r = await http.post(
                "/chat",
                json={"message": "search please", "enable_search": True},
                headers=headers,
            )
            assert r.status_code == 200, r.text
            assert "FastAPI Release Notes" in r.text
    finally:
        await persistence.stop()
