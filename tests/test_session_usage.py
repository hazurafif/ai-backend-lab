"""Offline tests for GET /threads/{id}/usage (session context + usage report).

Covers:

  - aggregation of usage_metadata (input/output/total tokens, run count)
  - context report: last run's input tokens vs the model context window
    (utilization + remaining), null window for unknown models
  - empty / unknown threads -> 404, ownership isolation
  - active_run flag from the run registry
"""

from __future__ import annotations

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
from app.core.constants import thread_metadata_ns
from app.core.run_registry import runs
from app.core.security import create_access_token
from app.main import create_app
from app.schema.agent_config_schema import AgentConfigIn
from app.services import agent_configs
from app.services.agent import AgentRegistry, build_backend
from app.util.date import now_iso

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


def make_app(persistence):
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
        model_factory=lambda m, t: _DummyModel(),
    )
    return create_app(agent_registry=registry)


class _DummyModel(BaseChatModel):
    """Minimal chat model returned by the model factory (never invoked)."""

    responses: list[AIMessage] = Field(default_factory=lambda: [AIMessage(content="ok")])
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "dummy"

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
        return self


async def client_for(app, username: str, role: str = "user") -> httpx.AsyncClient:
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def ai_message(
    message_id: str, *, input_tokens: int, output_tokens: int, content: str = "hi"
) -> dict:
    return {
        "id": message_id,
        "type": "ai",
        "content": content,
        "usage_metadata": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


async def seed_thread(
    username: str,
    thread_id: str,
    messages: list[dict],
    *,
    agent: str = "default",
) -> None:
    await database.persistence.chat_history.add_messages(thread_id, username, messages)
    await database.persistence.store.aput(
        thread_metadata_ns(username),
        thread_id,
        {"title": "Usage test", "agent": agent, "created_at": now_iso(), "updated_at": now_iso()},
    )


# ---------------------------------------------------------------------------
# usage + context aggregation
# ---------------------------------------------------------------------------


async def test_thread_usage_report(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        # a named agent config so the model resolves (gpt-4o family window)
        await agent_configs.create_config(
            persistence.store,
            AgentConfigIn(name="research", model="openai:gpt-4o-mini", description="x"),
            "alice",
            known_servers=[],
        )
        await seed_thread(
            "alice",
            "t1",
            [
                {"id": "u1", "type": "human", "content": "hello"},
                ai_message("a1", input_tokens=100, output_tokens=20, content="first answer"),
                {"id": "u2", "type": "human", "content": "tell me more"},
                ai_message("a2", input_tokens=250, output_tokens=40, content="second answer"),
            ],
            agent="research",
        )
        r = await client.get("/threads/t1/usage")
        assert r.status_code == 200, r.text
        body = r.json()

        # The AI SDK alias path (/api/chat/threads/...) returns the same payload.
        r_alias = await client.get("/api/chat/threads/t1/usage")
        assert r_alias.status_code == 200, r_alias.text
        assert r_alias.json() == body

        # the thread's agent + its model resolve from the agent config
        assert body["thread_id"] == "t1"
        assert body["agent"] == "research"
        assert body["model"] == "openai:gpt-4o-mini"  # default_spec model

        # messages: 4 stored, content length of "hello"+"first answer"+...
        assert body["messages"]["count"] == 4
        assert body["messages"]["characters"] > 0

        # cumulative usage: input summed per run (billed), output additive
        assert body["usage"] == {
            "input_tokens": 350,
            "output_tokens": 60,
            "total_tokens": 410,
            "runs": 2,
        }

        # context: last run's input (= current context) vs the model window
        ctx = body["context"]
        assert ctx["current_input_tokens"] == 250
        assert ctx["context_window"] == 128_000  # gpt-4o family
        assert ctx["utilization"] == round(250 / 128_000, 4)
        assert ctx["remaining_tokens"] == 128_000 - 250

        assert body["active_run"] is False


async def test_thread_usage_no_usage_metadata(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        await seed_thread(
            "alice",
            "t2",
            [
                {"id": "u1", "type": "human", "content": "hi"},
                {"id": "a1", "type": "ai", "content": "no usage reported"},
            ],
        )
        r = await client.get("/threads/t2/usage")
        body = r.json()
        assert body["usage"] is None
        assert body["context"] is None
        # builtin default agent -> model still resolves
        assert body["model"] == config.settings.model


async def test_thread_usage_unknown_model_window(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        await seed_thread(
            "alice",
            "t3",
            [ai_message("a1", input_tokens=10, output_tokens=5)],
            agent="weird",
        )
        # agent config "weird" doesn't exist -> model unknown -> window null
        r = await client.get("/threads/t3/usage")
        body = r.json()
        assert body["agent"] == "weird"
        assert body["model"] is None
        assert body["context"]["context_window"] is None
        assert body["context"]["utilization"] is None
        assert body["context"]["remaining_tokens"] is None
        assert body["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "runs": 1,
        }


# ---------------------------------------------------------------------------
# lifecycle + ownership
# ---------------------------------------------------------------------------


async def test_thread_usage_empty_and_unknown(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        # unknown thread -> 404 (no history, no metadata)
        assert (await client.get("/threads/ghost/usage")).status_code == 404
        # metadata-only thread (no messages yet) -> 404, like /messages
        await database.persistence.store.aput(
            thread_metadata_ns("alice"),
            "empty",
            {"title": "Empty", "created_at": now_iso(), "updated_at": now_iso()},
        )
        assert (await client.get("/threads/empty/usage")).status_code == 404


async def test_thread_usage_ownership(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app):
        async with await client_for(app, "alice") as alice:
            await seed_thread("alice", "t1", [ai_message("a1", input_tokens=10, output_tokens=5)])
            assert (await alice.get("/threads/t1/usage")).status_code == 200
        # bob cannot read alice's thread usage
        async with await client_for(app, "bob") as bob:
            assert (await bob.get("/threads/t1/usage")).status_code == 404


async def test_thread_usage_active_run(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        await seed_thread("alice", "t1", [ai_message("a1", input_tokens=10, output_tokens=5)])
        try:
            runs.register("t1")
            assert (await client.get("/threads/t1/usage")).json()["active_run"] is True
        finally:
            runs.unregister("t1")
