"""Offline tests for the new chat features:

- AI SDK HITL: interrupt surfaces as an `app.interrupt` custom chunk, resume
  via POST /api/chat with `decision`
- cancel: POST /threads/{id}/cancel + `done.cancelled` stream behavior
- thread CRUD: delete, rename, pagination, ownership 404s
- `usage` aggregation in the done event
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from test_smoke import Scripted, build_scripted_agent, echo, parse_sdk_data_lines, parse_sse_chunk

from app.core import config
from app.core.database import persistence
from app.core.run_registry import runs
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest_asyncio.fixture
async def memory_persistence():
    """Force in-memory checkpointer/store and start the app singleton."""
    config.settings.database_uri = None
    await persistence.start()
    yield persistence
    await persistence.stop()


def extract_done(text: str) -> dict:
    """Parse the last `done` event out of an SSE response body."""
    for block in text.split("\n\n"):
        if block.startswith("event: done"):
            return json.loads(block.split("\n", 1)[1].removeprefix("data: ").strip())
    raise AssertionError(f"no done event in response: {text[:200]!r}")


@tool
async def slow_tool(delay_s: float = 30) -> str:
    """Sleep for a while (used to keep a run in flight for cancel tests)."""
    await asyncio.sleep(delay_s)
    return "slow:done"


def build_slow_agent(checkpointer, store):
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-slow", name="slow_tool", args={"delay_s": 30})],
            ),
            AIMessage(content="Done after the slow tool."),
        ]
    )
    return build_agent(
        checkpointer=checkpointer,
        store=store,
        mcp_tools=[slow_tool],
        model=model,
        system_prompt="test",
    )


# ---------------------------------------------------------------------------
# AI SDK HITL: interrupt custom chunk + resume via /api/chat
# ---------------------------------------------------------------------------


async def test_ai_sdk_interrupt_emits_custom_chunk(memory_persistence):
    agent = build_scripted_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        interrupt_on={"echo": True},
    )
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        app.state.agent = agent
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "id": "chat-hitl",
                    "messages": [
                        {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "approve"}]}
                    ],
                },
            )
            assert r.status_code == 200, r.text
            chunks = parse_sdk_data_lines(r.text)

            # Interrupt surfaces as an app.interrupt custom chunk (not an error).
            interrupt = [c for c in chunks if c.get("kind") == "app.interrupt"]
            assert interrupt, [c["type"] for c in chunks]
            assert interrupt[0]["type"] == "custom"
            assert interrupt[0]["threadId"] == "chat-hitl"
            action = interrupt[0]["interrupts"][0]["action_requests"][0]
            assert action["name"] == "echo"
            assert action["args"] == {"x": "hello"}

            assert not any(c["type"] == "error" for c in chunks)
            finishes = [c for c in chunks if c["type"] == "finish"]
            assert finishes and finishes[-1]["finishReason"] == "other", chunks
            assert chunks[-1] == finishes[-1], "stream ends with finish"


async def test_ai_sdk_resume_via_decision(memory_persistence):
    agent = build_scripted_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        interrupt_on={"echo": True},
    )
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        app.state.agent = agent
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # First run pauses.
            r = await client.post(
                "/api/chat",
                json={
                    "id": "chat-hitl",
                    "messages": [
                        {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "x"}]}
                    ],
                },
            )
            assert any(c.get("kind") == "app.interrupt" for c in parse_sdk_data_lines(r.text))

            # Resume with a decision: tool executes, answer streams, done.
            r = await client.post(
                "/api/chat",
                json={"id": "chat-hitl", "decision": {"type": "approve"}, "messages": []},
            )
            assert r.status_code == 200, r.text
            chunks = parse_sdk_data_lines(r.text)
            assert any(c["type"] == "tool-output-available" for c in chunks), chunks
            text = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
            assert "Final answer from the agent." in text, text
            assert chunks[-1]["type"] == "finish" and chunks[-1]["finishReason"] == "stop"

            # The thread is no longer waiting -> 409.
            r = await client.post(
                "/api/chat",
                json={"id": "chat-hitl", "decision": {"type": "approve"}, "messages": []},
            )
            assert r.status_code == 409, r.text

            # Resume without id -> 422.
            r = await client.post(
                "/api/chat", json={"decision": {"type": "approve"}, "messages": []}
            )
            assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


async def test_cancel_aborts_running_stream(memory_persistence):
    agent = build_slow_agent(memory_persistence.checkpointer, memory_persistence.store)
    collected: list[tuple[str, dict]] = []

    async def collect() -> None:
        async for chunk in agent_stream(agent, "tester", message="go slow"):
            collected.append(parse_sse_chunk(chunk))

    task = asyncio.create_task(collect())
    # Wait until the run is registered (the thread_id is generated internally).
    thread_id = None
    for _ in range(200):
        ids = list(runs._stops)
        if ids:
            thread_id = ids[0]
            break
        await asyncio.sleep(0.01)
    assert thread_id is not None, "run was never registered"

    assert runs.cancel(thread_id)
    await asyncio.wait_for(task, timeout=10)

    done = [d for e, d in collected if e == "done"]
    assert done, "expected a terminal done event"
    assert done[-1].get("cancelled") is True, done[-1]
    assert not runs.is_running(thread_id), "run should be unregistered after cancel"


async def test_cancel_endpoint(memory_persistence):
    runs.register("thread-cancel-me")
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    try:
        async with app.router.lifespan_context(app):
            app.state.agent = build_scripted_agent(
                memory_persistence.checkpointer, memory_persistence.store
            )
            await memory_persistence.users.create_user(username="tester", hashed_password="x")
            token = create_access_token(data={"sub": "tester"})
            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post("/threads/thread-cancel-me/cancel", headers=headers)
                assert r.status_code == 200, r.text
                assert r.json()["status"] == "cancelled"

                # Second cancel: nothing running anymore.
                r = await client.post("/threads/thread-cancel-me/cancel", headers=headers)
                assert r.status_code == 409, r.text

                # A completed chat run is unregistered -> 409 too.
                r = await client.post("/chat", json={"message": "hi"}, headers=headers)
                tid = extract_done(r.text)["thread_id"]
                r = await client.post(f"/threads/{tid}/cancel", headers=headers)
                assert r.status_code == 409, r.text
    finally:
        runs.unregister("thread-cancel-me")


# ---------------------------------------------------------------------------
# thread CRUD + pagination + ownership
# ---------------------------------------------------------------------------


async def test_thread_crud_pagination_and_ownership(memory_persistence):
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        app.state.agent = build_scripted_agent(
            memory_persistence.checkpointer, memory_persistence.store
        )
        await memory_persistence.users.create_user(username="tester", hashed_password="x")
        await memory_persistence.users.create_user(username="bob", hashed_password="x")
        token = create_access_token(data={"sub": "tester"})
        bob_token = create_access_token(data={"sub": "bob"})
        headers = {"Authorization": f"Bearer {token}"}
        bob_headers = {"Authorization": f"Bearer {bob_token}"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Two chats -> two threads.
            r1 = await client.post("/chat", json={"message": "first thread"}, headers=headers)
            tid1 = extract_done(r1.text)["thread_id"]
            r2 = await client.post("/chat", json={"message": "second thread"}, headers=headers)
            tid2 = extract_done(r2.text)["thread_id"]
            assert tid1 != tid2

            # Pagination: newest first.
            r = await client.get("/threads?limit=1&offset=0", headers=headers)
            assert r.status_code == 200 and [t["thread_id"] for t in r.json()] == [tid2]
            r = await client.get("/threads?limit=1&offset=1", headers=headers)
            assert [t["thread_id"] for t in r.json()] == [tid1]
            r = await client.get("/threads?limit=0", headers=headers)
            assert r.status_code == 422, "limit must be >= 1"

            # Rename.
            r = await client.patch(f"/threads/{tid1}", json={"title": "renamed"}, headers=headers)
            assert r.status_code == 200 and r.json()["title"] == "renamed"
            r = await client.get("/threads", headers=headers)
            by_id = {t["thread_id"]: t for t in r.json()}
            assert by_id[tid1]["title"] == "renamed"

            # Ownership: bob cannot read/resume/delete tester's thread.
            r = await client.get(f"/threads/{tid1}/messages", headers=bob_headers)
            assert r.status_code == 404, r.text
            r = await client.post(
                f"/threads/{tid1}/resume",
                json={"decision": {"type": "approve"}},
                headers=bob_headers,
            )
            assert r.status_code == 404, r.text
            r = await client.delete(f"/threads/{tid1}", headers=bob_headers)
            assert r.status_code == 404, r.text

            # Delete: state + history + metadata gone.
            r = await client.delete(f"/threads/{tid1}", headers=headers)
            assert r.status_code == 204, r.text
            r = await client.get(f"/threads/{tid1}/messages", headers=headers)
            assert r.status_code == 404, r.text
            r = await client.get("/threads", headers=headers)
            assert [t["thread_id"] for t in r.json()] == [tid2]

            # Rename a missing thread -> 404.
            r = await client.patch("/threads/ghost", json={"title": "x"}, headers=headers)
            assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# usage aggregation in done
# ---------------------------------------------------------------------------


async def test_done_includes_usage(memory_persistence):
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-1", name="echo", args={"x": "hi"})],
            ),
            AIMessage(
                content="Final answer with usage.",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
        ]
    )
    agent = build_agent(
        checkpointer=memory_persistence.checkpointer,
        store=memory_persistence.store,
        mcp_tools=[echo],
        model=model,
        system_prompt="test",
    )

    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, "tester", message="usage please"):
        events.append(parse_sse_chunk(chunk))

    done = [d for e, d in events if e == "done"][-1]
    assert done["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, done
