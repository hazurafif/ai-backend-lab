"""Offline end-to-end tests — no API key, no network, no Postgres.

Uses a scripted fake chat model (same trick as deepagents' own tests) that
returns a fixed sequence of responses. Exercises the full pipeline:

    HTTP POST /chat  ->  agent astream_events(v3)  ->  SSE events
    interrupt + resume (human-in-the-loop)
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
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from app.core.config import settings
from app.core.database import persistence
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


# ---------------------------------------------------------------------------
# fixtures & helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memory_persistence():
    """Force in-memory checkpointer/store and start the app singleton."""
    settings.database_uri = None
    await persistence.start()
    yield persistence
    await persistence.stop()


@tool
def echo(x: str) -> str:
    """Echo the input back with a marker."""
    return f"echo:{x}"


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


def build_scripted_agent(checkpointer, store, *, interrupt_on: dict | None = None):
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-1", name="echo", args={"x": "hello"})],
            ),
            AIMessage(content="Final answer from the agent."),
        ]
    )
    return build_agent(
        checkpointer=checkpointer,
        store=store,
        mcp_tools=[echo],
        model=model,
        system_prompt="test",
        interrupt_on=interrupt_on,
    )


def build_reasoning_agent(checkpointer, store, *, reasoning: str, answer: str):
    """Agent whose model emits a reasoning block followed by the answer text."""
    model = Scripted(
        responses=[
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": reasoning},
                    {"type": "text", "text": answer},
                ]
            )
        ]
    )
    return build_agent(
        checkpointer=checkpointer,
        store=store,
        mcp_tools=[],
        model=model,
        system_prompt="test",
    )


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


def parse_sdk_data_lines(text: str) -> list[dict]:
    """Parse the `data:` lines of an AI SDK data-stream response."""
    chunks: list[dict] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line.removeprefix("data: ")
            if payload != "[DONE]":
                chunks.append(json.loads(payload))
    return chunks


async def collect_stream(agent, username, **kwargs) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, username, **kwargs):
        events.append(parse_sse_chunk(chunk))
    return events


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


async def test_direct_streaming_pipeline(memory_persistence):
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
    events = await collect_stream(agent, "tester", message="hi there")

    names = [e for e, _ in events]
    assert "message_delta" in names, "missing message_delta"
    assert "message" in names, "missing finalized message"
    assert "tool_start" in names, "missing tool_start"
    assert "tool_end" in names, "missing tool_end"
    done = [d for e, d in events if e == "done"]
    assert done and done[-1].get("messages"), "missing done with messages"
    assert done[-1].get("interrupted") is None, "should not be interrupted"

    # Chat history rows match the finalized messages of the done event.
    rows = await memory_persistence.chat_history.list_messages(done[-1]["thread_id"])
    assert [r["type"] for r in rows] == [m["type"] for m in done[-1]["messages"]]

    tool_end = next(d for e, d in events if e == "tool_end")
    assert tool_end["name"] == "echo"
    assert tool_end["output"]["content"] == "echo:hello", tool_end


async def test_reasoning_streaming_pipeline(memory_persistence):
    """Thinking content is streamed live as reasoning_delta events."""
    agent = build_reasoning_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        reasoning="Let me reason step by step: the answer is 42.",
        answer="The answer is 42.",
    )
    events = await collect_stream(agent, "tester", message="think hard")

    names = [e for e, _ in events]
    assert "reasoning_start" in names, "missing reasoning_start"
    assert "reasoning_delta" in names, "missing reasoning_delta events"
    assert "reasoning_end" in names, "missing reasoning_end"
    assert "message_delta" in names, "missing message_delta events"

    # Thinking is a bracketed lifecycle (start -> delta* -> end), shaped
    # like a tool call, with a stable id across the whole turn.
    reasoning = [d for e, d in events if e == "reasoning_delta"]
    assert "".join(d["delta"] for d in reasoning) == "Let me reason step by step: the answer is 42."
    first_idx = names.index("reasoning_start")
    deltas = [i for i, e in enumerate(names) if e == "reasoning_delta"]
    end_idx = names.index("reasoning_end")
    assert deltas and first_idx < deltas[0] and deltas[-1] < end_idx
    starts = [d for e, d in events if e == "reasoning_start"]
    ends = [d for e, d in events if e == "reasoning_end"]
    assert len(starts) == len(ends) == 1
    assert starts[0]["id"] == ends[0]["id"] == reasoning[0]["id"]
    text = "".join(d["delta"] for e, d in events if e == "message_delta")
    assert text == "The answer is 42."
    msg_idx = names.index("message")
    assert end_idx < msg_idx

    # The finalized message keeps the reasoning block (langchain schema).
    msg = next(d for e, d in events if e == "message")
    content = msg["message"]["content"]
    assert content[0]["type"] == "reasoning"
    assert content[0]["reasoning"] == "Let me reason step by step: the answer is 42."
    assert content[1]["type"] == "text"

    # The stored thread history preserves the reasoning block too.
    done = [d for e, d in events if e == "done"]
    stored = await memory_persistence.chat_history.list_messages(done[-1]["thread_id"])
    assert stored and stored[-1]["content"][0]["type"] == "reasoning"


async def test_interrupt_and_resume(memory_persistence):
    agent = build_scripted_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        interrupt_on={"echo": True},
    )

    # Run 1: the tool call is paused for human approval.
    events = await collect_stream(agent, "tester", message="approve this tool call")
    names = [e for e, _ in events]
    assert "interrupt" in names, "missing interrupt event"

    interrupt = next(d for e, d in events if e == "interrupt")
    action_requests = interrupt["interrupts"][0]["action_requests"]
    assert action_requests[0]["name"] == "echo", action_requests
    assert action_requests[0]["args"] == {"x": "hello"}

    done = [d for e, d in events if e == "done"][-1]
    assert done["interrupted"] is True
    assert not any(e == "tool_start" for e in names), "tool must NOT run before approval"
    thread_id = done["thread_id"]

    # Resume with approval -> tool executes, final answer streams.
    events2 = await collect_stream(
        agent, "tester", thread_id=thread_id, resume={"decisions": [{"type": "approve"}]}
    )
    names2 = [e for e, _ in events2]
    assert "tool_start" in names2 and "tool_end" in names2, "tool should run after approval"
    done2 = [d for e, d in events2 if e == "done"][-1]
    assert not done2.get("interrupted"), "run should complete after resume"

    def extract_text(m: dict) -> str:
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(b.get("text", "") for b in c if b.get("type") == "text")
        return ""

    texts = [extract_text(m) for m in done2["messages"] if m.get("type") == "ai"]
    assert any("Final answer" in t for t in texts), texts

    # Chat history: rows written at interrupt (human + tool-call AI) and the
    # resumed run appended the rest — no duplicates across the two runs.
    rows = await memory_persistence.chat_history.list_messages(thread_id)
    assert [r["type"] for r in rows] == [m["type"] for m in done2["messages"]]
    assert len(rows) == len(done2["messages"]) == 4, rows
    assert len({r.get("id") for r in rows}) == len(rows), "duplicate message ids"


async def test_ai_sdk_extract_user_message():
    from app.services.ai_sdk_chat import extract_user_message

    # parts format (what useChat sends)
    parts = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hello via sdk"}]},
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hi"}]},
        {"id": "u2", "role": "user", "parts": [{"type": "text", "text": "second msg"}]},
    ]
    assert extract_user_message(parts) == "second msg"
    # legacy string content
    assert extract_user_message([{"role": "user", "content": "plain"}]) == "plain"
    # empty / no user message
    assert extract_user_message([]) == ""
    assert extract_user_message([{"role": "assistant", "content": "x"}]) == ""


async def test_ai_sdk_chat_endpoint(memory_persistence):
    """POST /api/chat speaks the AI SDK data-stream protocol (useChat)."""
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        _agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
        app.state.agent = _agent
        app.state.agents.set_static_default(_agent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "id": "chat-abc",
                    "selectedChatModel": "gpt-4o-mini",
                    "messages": [
                        {
                            "id": "m1",
                            "role": "user",
                            "parts": [{"type": "text", "text": "hello via sdk"}],
                        }
                    ],
                },
            )
            assert r.status_code == 200, r.text

            chunks = parse_sdk_data_lines(r.text)
            types = [c["type"] for c in chunks]
            assert types[0] == "start", types
            assert "text-start" in types, "missing text-start"
            assert "text-delta" in types, "missing text-delta"
            assert "text-end" in types, "missing text-end"
            assert types[-1] == "finish", types
            assert chunks[-1]["finishReason"] == "stop"
            assert "data: [DONE]" in r.text

            # tool activity surfaces as native AI SDK tool chunks (echo tool in scripted agent)
            tool_input = [c for c in chunks if c["type"] == "tool-input-start"]
            assert tool_input, [c["type"] for c in chunks]
            assert tool_input[0]["toolName"] == "echo"
            assert any(c["type"] == "tool-input-available" for c in chunks)
            assert any(c["type"] == "tool-output-available" for c in chunks)

            tool_available = next(c for c in chunks if c["type"] == "tool-input-available")
            assert tool_available["input"] == {"x": "hello"}, tool_available

            # text of the answer is streamed verbatim
            text = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
            assert "Final answer from the agent." in text, text

            # no user message -> 422
            r = await client.post("/api/chat", json={"id": "x", "messages": []})
            assert r.status_code == 422, r.text


async def test_chat_history_persisted_without_duplicates(memory_persistence):
    """Every turn appends exactly its messages to the chat history table."""
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)

    done1 = [
        d for e, d in await collect_stream(agent, "tester", message="turn one") if e == "done"
    ][-1]
    thread_id = done1["thread_id"]
    rows = await memory_persistence.chat_history.list_messages(thread_id)
    assert [r["type"] for r in rows] == [m["type"] for m in done1["messages"]]

    # Second turn on the same thread: new messages appended, old ones kept.
    done2 = [
        d
        for e, d in await collect_stream(agent, "tester", thread_id=thread_id, message="turn two")
        if e == "done"
    ][-1]
    rows = await memory_persistence.chat_history.list_messages(thread_id)
    assert [r["type"] for r in rows] == [m["type"] for m in done2["messages"]]
    assert len(rows) == len(done2["messages"])
    assert len({r.get("id") for r in rows}) == len(rows), "duplicate message ids"

    # Thread metadata: title from the first message, updated_at refreshed.
    item = await memory_persistence.store.aget(("threads", "tester"), thread_id)
    assert item.value["title"] == "turn one"
    assert item.value["updated_at"] >= item.value["created_at"]


async def test_http_end_to_end(memory_persistence):
    # Placeholder agent so lifespan skips the real (OpenAI) model; the
    # scripted agent is built on the SAME persistence instances the API uses.
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        _agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
        app.state.agent = _agent
        app.state.agents.set_static_default(_agent)
        # get_current_user validates against the users store.
        await memory_persistence.users.create_user(username="tester", hashed_password="x")
        token = create_access_token(data={"sub": "tester"})

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # auth required
            r = await client.get("/threads")
            assert r.status_code == 401

            headers = {"Authorization": f"Bearer {token}"}
            r = await client.post("/chat", json={"message": "hello via http"}, headers=headers)
            assert r.status_code == 200, r.text
            assert "event: message_delta" in r.text
            assert "event: done" in r.text

            r = await client.get("/threads", headers=headers)
            assert r.status_code == 200
            threads = r.json()
            assert len(threads) >= 1 and threads[0]["title"] == "hello via http", threads

            thread_id = threads[0]["thread_id"]
            r = await client.get(f"/threads/{thread_id}/messages", headers=headers)
            assert r.status_code == 200 and len(r.json()) >= 2, r.text

            # Legacy threads (no rows in the chat_messages table) fall back to
            # checkpoint rehydration — same response shape.
            memory_persistence.chat_history._memory.clear()
            memory_persistence.chat_history._memory_ids.clear()
            r = await client.get(f"/threads/{thread_id}/messages", headers=headers)
            assert r.status_code == 200 and len(r.json()) >= 2, r.text

            # resume on a non-interrupted thread -> 409
            r = await client.post(
                f"/threads/{thread_id}/resume",
                json={"decision": {"type": "approve"}},
                headers=headers,
            )
            assert r.status_code == 409, r.text
