"""Offline tests for LLM-generated thread titles (POST /threads/{id}/title).

Covers the prompt-template rendering, the deterministic fallback, and the
endpoint: generation via the thread's agent model, metadata upsert (create +
update), ownership scoping and the no-messages 404.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from app.core import config, database
from app.core.constants import thread_metadata_ns
from app.core.security import create_access_token
from app.main import create_app
from app.services import title_generator
from app.services.agent import AgentRegistry, build_backend

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class TitleModel(BaseChatModel):
    """Scripted model: chat turns and title-generation calls get separate queues."""

    responses: list[AIMessage] = Field(default_factory=lambda: [AIMessage(content="ok")])
    title_responses: list[AIMessage] = Field(
        default_factory=lambda: [AIMessage(content="generated title")]
    )
    prompts: list[list[Any]] = Field(default_factory=list)
    _chat_idx: int = 0
    _title_idx: int = 0
    tools: Sequence[dict | type] = ()

    @property
    def _llm_type(self) -> str:
        return "title"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        is_title = any(
            getattr(m, "type", "") == "system" and "title generator" in str(m.content)
            for m in messages
        )
        if is_title:
            seq = self.title_responses
            i = min(self._title_idx, len(seq) - 1)
            self._title_idx += 1
        else:
            seq = self.responses
            i = min(self._chat_idx, len(seq) - 1)
            self._chat_idx += 1
        return ChatResult(generations=[ChatGeneration(message=seq[i])])

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseChatModel],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = tools
        return self


@pytest_asyncio.fixture
async def memory_persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


def _make_app(model: TitleModel):
    """App whose agent registry builds every model from `model` (chat + title)."""
    registry = AgentRegistry(
        checkpointer=database.persistence.checkpointer,
        store=database.persistence.store,
        backend=build_backend(store=database.persistence.store),
        model_factory=lambda m, t: model,
    )
    return create_app(agent_registry=registry), model


async def _client(app, username: str, role: str = "user") -> httpx.AsyncClient:
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


async def _seed_thread(client: httpx.AsyncClient) -> str:
    """Run one chat turn and return the created thread id."""
    import json

    async with client.stream("POST", "/chat", json={"message": "how do I deploy fastapi?"}) as resp:
        text = "".join([chunk async for chunk in resp.aiter_text()])
    done = [b for b in text.split("\n\n") if b.startswith("event: done")]
    return json.loads(done[0].split("\n", 1)[1].removeprefix("data: ").strip())["thread_id"]


# ---------------------------------------------------------------------------
# template + fallback units
# ---------------------------------------------------------------------------


def test_format_conversation_renders_latest_messages():
    messages = [
        HumanMessage(content="first question"),
        AIMessage(content="first answer"),
        HumanMessage(content="second question"),
    ]
    text = title_generator.format_conversation(messages)
    assert text == "user: first question\nassistant: first answer\nuser: second question"


def test_format_conversation_skips_reasoning_blocks():
    msg = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "hmm"},
            {"type": "text", "text": "the answer"},
        ]
    )
    assert title_generator.format_conversation([msg]) == "assistant: the answer"


def test_clean_title():
    assert (
        title_generator.clean_title('  "deploy fastapi on fly.io".  ') == "deploy fastapi on fly.io"
    )
    assert title_generator.clean_title("x") == ""
    assert title_generator.clean_title("a b c") == "a b c"


async def test_fallback_title_uses_first_user_message():
    messages = [HumanMessage(content="  how do I deploy fastapi on fly.io?  ")]
    assert await title_generator.generate_title("openai:gpt-4o-mini", messages) == (
        "how do I deploy fastapi on fly.io?"
    )


async def test_generate_title_uses_template():
    model = TitleModel(title_responses=[AIMessage(content='"fix postgres connection errors"')])
    title = await title_generator.generate_title(model, [HumanMessage(content="conn refused")])
    assert title == "fix postgres connection errors"
    # The model saw the template with the rendered conversation.
    system = model.prompts[-1][0].content
    assert "title generator" in system
    assert "user: conn refused" in system


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------


async def test_title_endpoint_creates_and_upserts(memory_persistence):
    model = TitleModel(
        responses=[AIMessage(content="ok")],
        title_responses=[
            AIMessage(content="deploy fastapi"),
            AIMessage(content="scale fastapi"),
        ],
    )
    app, model = _make_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        thread_id = await _seed_thread(client)

        r = await client.post(f"/threads/{thread_id}/title")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["title"] == "deploy fastapi"
        assert body["thread_id"] == thread_id

        # Upsert: a second call updates the title in place.
        r = await client.post(f"/threads/{thread_id}/title")
        assert r.json()["title"] == "scale fastapi"

        # The thread list reflects the generated title.
        threads = (await client.get("/threads")).json()
        entry = next(t for t in threads if t["thread_id"] == thread_id)
        assert entry["title"] == "scale fastapi"


async def test_title_endpoint_creates_missing_metadata(memory_persistence):
    """Legacy threads without metadata rows still get titled."""
    model = TitleModel(title_responses=[AIMessage(content="legacy thread")])
    app, model = _make_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        thread_id = await _seed_thread(client)
        # Drop the metadata row, keep the checkpoint messages.
        await database.persistence.store.adelete(thread_metadata_ns("alice"), thread_id)
        assert await database.persistence.store.aget(thread_metadata_ns("alice"), thread_id) is None

        r = await client.post(f"/threads/{thread_id}/title")
        assert r.status_code == 200, r.text
        assert r.json()["title"] == "legacy thread"
        assert (
            await database.persistence.store.aget(thread_metadata_ns("alice"), thread_id)
            is not None
        )


async def test_title_endpoint_ownership_and_errors(memory_persistence):
    model = TitleModel()
    app, model = _make_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as alice:
        thread_id = await _seed_thread(alice)
        # Another user cannot title it.
        async with await _client(app, "bob") as bob:
            r = await bob.post(f"/threads/{thread_id}/title")
            assert r.status_code == 404
        # Unknown thread -> 404 (no checkpoint messages yet).
        r = await alice.post("/threads/nope/title")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# follow-up endpoint (frontend: call after each run)
# ---------------------------------------------------------------------------


async def test_followup_generates_when_title_is_default_truncation(memory_persistence):
    """Fresh threads carry the raw first-message truncation -> LLM title."""
    model = TitleModel(title_responses=[AIMessage(content="deploy fastapi")])
    app, model = _make_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        thread_id = await _seed_thread(client)
        # The chat service set the truncated default title.
        meta = await database.persistence.store.aget(thread_metadata_ns("alice"), thread_id)
        assert meta.value["title"] == "how do I deploy fastapi?"

        r = await client.post(f"/threads/{thread_id}/followup")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {"thread_id": thread_id, "title": "deploy fastapi", "generated": True}


async def test_followup_keeps_intentional_title_without_llm_call(memory_persistence):
    """A second call (or an already-titled thread) spends no tokens."""
    model = TitleModel(
        title_responses=[
            AIMessage(content="deploy fastapi"),
            AIMessage(content="regenerated title"),
        ]
    )
    app, model = _make_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        thread_id = await _seed_thread(client)

        r = await client.post(f"/threads/{thread_id}/followup")
        assert r.json()["generated"] is True
        calls_after = len(model.prompts)

        # Intentional title now -> no regeneration, no model call.
        r = await client.post(f"/threads/{thread_id}/followup")
        body = r.json()
        assert body == {"thread_id": thread_id, "title": "deploy fastapi", "generated": False}
        assert len(model.prompts) == calls_after, "LLM called despite an intentional title"

        # force: true regenerates.
        r = await client.post(f"/threads/{thread_id}/followup", json={"force": True})
        body = r.json()
        assert body["title"] == "regenerated title" and body["generated"] is True


async def test_followup_creates_metadata_and_404s(memory_persistence):
    model = TitleModel(title_responses=[AIMessage(content="legacy thread")])
    app, model = _make_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        thread_id = await _seed_thread(client)
        await database.persistence.store.adelete(thread_metadata_ns("alice"), thread_id)

        # No metadata row -> generated (row created).
        r = await client.post(f"/threads/{thread_id}/followup")
        assert r.json() == {
            "thread_id": thread_id,
            "title": "legacy thread",
            "generated": True,
        }
        assert (
            await database.persistence.store.aget(thread_metadata_ns("alice"), thread_id)
            is not None
        )

        # Unknown thread -> 404.
        r = await client.post("/threads/nope/followup")
        assert r.status_code == 404
