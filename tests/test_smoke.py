"""Offline end-to-end tests — no API key, no network, no Postgres.

Uses a scripted fake chat model (same trick as deepagents' own tests) that
returns a fixed sequence of responses. Exercises the full pipeline:

    HTTP POST /chat  ->  agent astream_events(v3)  ->  SSE events
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
from app.schemas import ChatRequest

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


def build_scripted_agent(checkpointer, store):
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
    )


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


async def collect_stream(request: ChatRequest, username: str, agent) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for chunk in _agent_stream(request, username, agent):
        events.append(parse_sse_chunk(chunk))
    return events


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


async def test_direct_streaming_pipeline(memory_persistence):
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
    events = await collect_stream(ChatRequest(message="hi there"), "tester", agent)

    names = [e for e, _ in events]
    assert "message_delta" in names, "missing message_delta"
    assert "message" in names, "missing finalized message"
    assert "tool_start" in names, "missing tool_start"
    assert "tool_end" in names, "missing tool_end"
    done = [d for e, d in events if e == "done"]
    assert done and done[-1].get("messages"), "missing done with messages"

    tool_end = next(d for e, d in events if e == "tool_end")
    assert tool_end["name"] == "echo"
    assert tool_end["output"]["content"] == "echo:hello", tool_end


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
