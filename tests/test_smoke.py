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

from app import auth, db
from app.agent import build_agent
from app.config import settings
from app.main import _agent_stream, create_app

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
    await db.persistence.start()
    yield db.persistence
    await db.persistence.stop()


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
    async for chunk in _agent_stream(agent, username, **kwargs):
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

    tool_end = next(d for e, d in events if e == "tool_end")
    assert tool_end["name"] == "echo"
    assert tool_end["output"]["content"] == "echo:hello", tool_end


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


async def test_ai_sdk_extract_user_message():
    from app.ai_sdk_chat import extract_user_message

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
        app.state.agent = build_scripted_agent(
            memory_persistence.checkpointer, memory_persistence.store
        )
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

            # tool activity surfaces as custom chunks (echo tool in scripted agent)
            custom = [c for c in chunks if c["type"] == "custom"]
            kinds = [c["kind"] for c in custom]
            assert "tool-start" in kinds and "tool-end" in kinds, kinds
            tool_start = next(c for c in custom if c["kind"] == "tool-start")
            assert tool_start["providerMetadata"]["name"] == "echo"

            # text of the answer is streamed verbatim
            text = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
            assert "Final answer from the agent." in text, text

            # no user message -> 422
            r = await client.post("/api/chat", json={"id": "x", "messages": []})
            assert r.status_code == 422, r.text


async def test_http_end_to_end(memory_persistence):
    # Placeholder agent so lifespan skips the real (OpenAI) model; the
    # scripted agent is built on the SAME persistence instances the API uses.
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        app.state.agent = build_scripted_agent(
            memory_persistence.checkpointer, memory_persistence.store
        )
        token = auth.create_access_token(data={"sub": "tester"})

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

            # resume on a non-interrupted thread -> 409
            r = await client.post(
                f"/threads/{thread_id}/resume",
                json={"decision": {"type": "approve"}},
                headers=headers,
            )
            assert r.status_code == 409, r.text
