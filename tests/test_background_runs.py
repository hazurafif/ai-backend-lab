"""Background-run behavior: runs survive client disconnects ("new chat"),
persist history while running, and emit lifecycle notifications.

Note on transport: the offline suite uses httpx.ASGITransport, which buffers
responses until the ASGI app completes — open-ended SSE streams can't be
exercised over HTTP here. Disconnect/attach/cancel flows therefore drive the
real streaming generators directly (the exact code paths the HTTP routes
use), while HTTP covers the endpoints that complete (conflict 409s, thread
messages, notifications list).

Covers:
- closing the stream mid-run -> run still completes, history + metadata written
- explicit cancel -> aborts, partial history saved, run_cancelled emitted
- notification stream + REST list -> run_started + run_completed events
- attach to an active run; 409 when idle or when a second run starts
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, ToolCall
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from test_chat_features import slow_tool
from test_smoke import Scripted, build_scripted_agent, collect_stream, parse_sse_chunk

from app.core import config
from app.core.constants import thread_metadata_ns
from app.core.database import persistence
from app.core.notification_hub import hub
from app.core.run_manager import run_manager
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream, attach_stream, notification_stream

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest_asyncio.fixture
async def memory_persistence():
    """Force in-memory checkpointer/store and reset the notification hub."""
    config.settings.database_uri = None
    hub.reset()
    await persistence.start()
    yield persistence
    await persistence.stop()


def build_slow_agent(checkpointer, store, delay_s: float = 0.3):
    """Agent that calls a sleeping tool, then answers (keeps runs in flight)."""
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-slow", name="slow_tool", args={"delay_s": delay_s})],
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


async def make_client(app, username: str = "tester") -> httpx.AsyncClient:
    await persistence.users.create_user(username=username, hashed_password="x")
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


async def wait_until(predicate) -> None:
    for _ in range(200):  # up to 10s
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met within timeout")


async def test_run_continues_after_stream_close(memory_persistence):
    """Closing the SSE stream (clicking "new chat") does not abort the run."""
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        agent = build_slow_agent(
            memory_persistence.checkpointer, memory_persistence.store, delay_s=0.3
        )
        app.state.agent = agent

        # Start a chat and walk away right after the first event arrives
        # (closing the generator is what a client disconnect does).
        stream = agent_stream(agent, "tester", message="go", thread_id="bg-1")
        async for _chunk in stream:
            break

        # The run keeps processing in the background and completes.
        await wait_until(lambda: not run_manager.is_running("bg-1"))
        item = await persistence.store.aget(thread_metadata_ns("tester"), "bg-1")
        assert item is not None and item.value.get("status") == "completed", item

        # History was written while nobody was connected.
        history = await persistence.chat_history.list_messages("bg-1")
        assert any(m.get("type") == "ai" for m in history), history

        # Lifecycle events were published for the user.
        types = [e["type"] for e in hub.recent("tester")]
        assert types == ["run_completed", "run_started"], types

        # The finished thread is served by the API like any other.
        async with await make_client(app) as client:
            r = await client.get("/threads/bg-1/messages")
            assert r.status_code == 200, r.text
            assert any(m.get("type") == "ai" for m in r.json()), r.text


async def test_cancel_aborts_but_saves_partial_history(memory_persistence):
    """Explicit cancel stops the run, yet finalized messages stay in history."""
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        agent = build_slow_agent(
            memory_persistence.checkpointer, memory_persistence.store, delay_s=30
        )
        app.state.agent = agent
        collected: list[tuple[str, dict]] = []

        async def consume() -> None:
            async for chunk in agent_stream(agent, "tester", message="go", thread_id="bg-cancel"):
                collected.append(parse_sse_chunk(chunk))

        task = asyncio.create_task(consume())
        # Wait until the tool-call message finalized (it is persisted at that
        # point), then abort the run through the real HTTP endpoint.
        await wait_until(lambda: any(e == "message" for e, _ in collected))
        async with await make_client(app) as client:
            r = await client.post("/threads/bg-cancel/cancel")
            assert r.status_code == 200, r.text
        await asyncio.wait_for(task, timeout=10)

        done = [d for e, d in collected if e == "done"]
        assert done and done[-1].get("cancelled") is True, done
        assert not run_manager.is_running("bg-cancel")

        # Partial history survives the abort (the tool-call message).
        history = await persistence.chat_history.list_messages("bg-cancel")
        assert any(m.get("type") == "ai" for m in history), history

        item = await persistence.store.aget(thread_metadata_ns("tester"), "bg-cancel")
        assert item is not None and item.value.get("status") == "cancelled", item
        types = [e["type"] for e in hub.recent("tester")]
        assert types == ["run_cancelled", "run_started"], types


async def test_notification_stream_and_list(memory_persistence):
    """The notification stream delivers run_started + run_completed."""
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        agent = build_slow_agent(
            memory_persistence.checkpointer, memory_persistence.store, delay_s=0.2
        )
        app.state.agent = agent
        chunks: list[str] = []

        async def watch() -> None:
            async for chunk in notification_stream("tester"):
                chunks.append(chunk)
                if "run_completed" in chunk:
                    return

        watcher = asyncio.create_task(watch())
        await asyncio.sleep(0.05)  # let the subscription attach
        events = await collect_stream(agent, "tester", message="hi")
        thread_id = [d for e, d in events if e == "done"][-1]["thread_id"]
        await asyncio.wait_for(watcher, timeout=10)

        parsed = [parse_sse_chunk(c) for c in chunks]
        lifecycle = [d for e, d in parsed if e.startswith("run_")]
        types = [d["type"] for d in lifecycle]
        assert types == ["run_started", "run_completed"], types
        completed = lifecycle[-1]
        assert completed["thread_id"] == thread_id
        assert completed["status"] == "completed"
        assert completed["title"] == "hi"

        # REST view agrees (newest first).
        async with await make_client(app) as client:
            r = await client.get("/notifications")
            assert r.status_code == 200, r.text
            payload = r.json()
            assert payload[0]["thread_id"] == thread_id
            assert payload[0]["type"] == "run_completed"


async def test_attach_stream_and_same_thread_conflict(memory_persistence):
    """Attaching watches an active run; a second run on the same thread is 409."""
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))
    async with app.router.lifespan_context(app):
        agent = build_slow_agent(
            memory_persistence.checkpointer, memory_persistence.store, delay_s=1.0
        )
        app.state.agent = agent
        # A first client starts a chat and keeps consuming it in the background.
        chat_task = asyncio.create_task(
            collect_stream(agent, "tester", message="go", thread_id="bg-attach")
        )
        await wait_until(lambda: run_manager.get("bg-attach") is not None)

        async with await make_client(app) as client:
            # A second chat on the same thread conflicts while the run is live.
            r = await client.post("/chat", json={"message": "again", "thread_id": "bg-attach"})
            assert r.status_code == 409, r.text

        # Attach to the active run and watch the answer stream live.
        deltas: list[str] = []
        async for chunk in attach_stream("bg-attach"):
            ev, data = parse_sse_chunk(chunk)
            if ev == "message_delta":
                deltas.append(data["delta"])
            if ev == "done":
                break
        assert "".join(deltas) == "Done after the slow tool.", deltas
        await asyncio.wait_for(chat_task, timeout=10)

        # Idle thread: attaching returns 409.
        async with await make_client(app) as client:
            r = await client.get("/threads/bg-attach/stream")
            assert r.status_code == 409, r.text
