"""Chat service: runs the agent and normalizes the v3 stream to SSE events.

`agent_stream` starts a background run (decoupled from the HTTP stream) and
returns a live SSE subscription to it; it is the single entry point used by
the `/chat` and `/api/chat` routers (and the AI SDK bridge). The SSE event
contract below is normative — keep it in sync with the frontend consumers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..core.constants import SSE_HEADERS, thread_metadata_ns
from ..core.database import persistence
from ..core.exceptions import Conflict
from ..core.notification_hub import hub
from ..core.run_manager import ActiveRun, run_manager
from ..core.run_registry import runs
from ..util.date import now_iso

logger = logging.getLogger(__name__)


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field from an object or dict, trying several names."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        else:
            try:
                v = getattr(obj, n, None)
            except Exception:
                v = None
            if v is not None:
                return v
    return default


def _serialize_message(msg: Any) -> dict:
    try:
        return msg.model_dump(mode="json", exclude_none=True)
    except Exception:
        return {
            "type": getattr(msg, "type", "unknown"),
            "content": str(getattr(msg, "content", "")),
        }


def _jsonable(v: Any) -> Any:
    """Coerce arbitrary values (messages, tool outputs, ...) to JSON-safe data."""
    if v is None or isinstance(v, str | int | float | bool):
        return v
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    return str(v)


def _aggregate_usage(messages: list[dict]) -> dict | None:
    """Sum usage_metadata (input/output/total tokens) over finalized messages."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = False
    for m in messages:
        usage = m.get("usage_metadata") if isinstance(m, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
                seen = True
    return totals if seen else None


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\ndata: {json.dumps(_jsonable(data), ensure_ascii=False, default=str)}\n\n"
    )


async def _record_thread_metadata(
    thread_id: str,
    username: str,
    message: str | None,
    agent_name: str = "default",
    *,
    status: str = "running",
) -> dict:
    """Upsert thread metadata: title on first message, `status` + `updated_at` on every call.

    Called at run start (so the thread shows up in GET /threads immediately,
    marked `running`) and at every terminal transition. Returns the stored
    value dict (used for notification payloads).
    """
    now = now_iso()
    value: dict = {}
    try:
        ns = thread_metadata_ns(username)
        existing = await persistence.store.aget(ns, thread_id)
        value = dict(existing.value) if existing is not None else {}
        if message is not None:
            value.setdefault("title", message[:80])
            value.setdefault("created_at", now)
        value["updated_at"] = now
        value["agent"] = agent_name
        value["status"] = status
        await persistence.store.aput(ns, thread_id, value)
    except Exception:
        logger.exception("failed to record thread metadata for %s", thread_id)
    return value


async def _save_history(thread_id: str, username: str, messages: list[dict]) -> None:
    """Persist finalized messages to the chat history table (best-effort).

    Deduped by message id, so resume/retry runs never duplicate rows.
    """
    if not messages:
        return
    try:
        added = await persistence.chat_history.add_messages(thread_id, username, messages)
        if added:
            logger.info("chat history: %d new messages for thread %s", added, thread_id)
    except Exception:
        logger.exception("failed to save chat history for thread %s", thread_id)


def _collect_interrupts(snapshot: Any) -> list[Any]:
    """Extract interrupt payloads from a state snapshot's pending tasks."""
    out: list[Any] = []
    for task in getattr(snapshot, "tasks", None) or []:
        for it in getattr(task, "interrupts", None) or []:
            if hasattr(it, "value"):  # langgraph Interrupt dataclass
                out.append(it.value)
            elif isinstance(it, dict):
                out.append(it.get("value"))
            else:
                out.append(it)
    return out


def sse_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------------------------------------------------------------------
# background runs: the run outlives its HTTP stream
# ---------------------------------------------------------------------------


def _run_event(status: str, thread_id: str, metadata: dict | None, agent_name: str) -> dict:
    """Lifecycle event payload for the per-user notification hub."""
    return {
        "type": f"run_{status}",
        "thread_id": thread_id,
        "title": (metadata or {}).get("title"),
        "agent": agent_name,
        "status": status,
        "at": now_iso(),
    }


def agent_stream(
    agent: CompiledStateGraph,
    username: str,
    *,
    message: str | None = None,
    thread_id: str | None = None,
    resume: Any = None,
    agent_name: str = "default",
) -> AsyncIterator[str]:
    """Start a background agent run and return its live SSE stream.

    The run executes in its own asyncio task, decoupled from the HTTP
    response: closing the stream (client disconnect, "new chat") leaves the
    run running — messages keep being persisted to chat history and a
    lifecycle event lands in the user's notification stream. Only
    `POST /threads/{id}/cancel` aborts a run; starting a second run on a
    thread with an active one raises Conflict (HTTP 409).

    - `message`: new user message for /chat
    - `resume`: value for `Command(resume=...)` (HITL continuation, /resume)
    - `agent_name`: agent config this thread runs on (recorded in thread metadata)
    """
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    active = run_manager.start(thread_id, username, agent_name)
    queue = run_manager.subscribe(active)
    active.task = asyncio.create_task(
        _run_agent(
            active,
            agent,
            username,
            message=message,
            thread_id=thread_id,
            resume=resume,
            agent_name=agent_name,
        )
    )
    return _stream_subscription(active, queue)


def attach_stream(thread_id: str) -> AsyncIterator[str]:
    """Attach a live SSE stream to an already-running thread.

    Raises Conflict when the thread has no active run (or it just finished) —
    the frontend falls back to GET /threads/{id}/messages in that case.
    """
    active = run_manager.get(thread_id)
    if active is None or active.done.is_set():
        raise Conflict(detail=f"Thread '{thread_id}' has no active run")
    queue = run_manager.subscribe(active)
    return _stream_subscription(active, queue)


async def _stream_subscription(active: ActiveRun, queue: asyncio.Queue) -> AsyncIterator[str]:
    """Drain one subscriber queue into SSE chunks until the run finishes.

    Detaching (generator close, client disconnect) only unsubscribes — the
    run keeps going in the background.
    """
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done_task = asyncio.create_task(active.done.wait())
            done_tasks, pending = await asyncio.wait(
                {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if done_task not in done_tasks:
                item = get_task.result()
                yield _sse(item["event"], item["data"])
                continue
            # Run finished: drain the terminal events queued before `done` set.
            if get_task in done_tasks:
                item = get_task.result()
                yield _sse(item["event"], item["data"])
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                yield _sse(item["event"], item["data"])
    finally:
        run_manager.unsubscribe(active, queue)


async def notification_stream(username: str, since: int | None = None) -> AsyncIterator[str]:
    """Per-user run lifecycle events as SSE (`run_started` ... `run_failed`).

    One long-lived connection per client. The hub replays recent events on
    connect; `since` (an event seq) skips already-seen ones. A comment line is
    emitted every 15s to keep the connection alive.
    """
    queue = hub.subscribe(username, since=since)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            yield _sse(event["type"], event)
    finally:
        hub.unsubscribe(username, queue)


async def _run_agent(
    active: ActiveRun,
    agent: CompiledStateGraph,
    username: str,
    *,
    message: str | None,
    thread_id: str,
    resume: Any,
    agent_name: str,
) -> None:
    """Background run body: stream the agent, persist, notify. Never raises."""
    config = {"configurable": {"thread_id": thread_id}}

    # Show the thread in history immediately, marked as running.
    metadata = await _record_thread_metadata(
        thread_id, username, message, agent_name, status="running"
    )
    hub.publish(username, _run_event("started", thread_id, metadata, agent_name))

    if resume is not None:
        run_input: Any = Command(resume=resume)
    else:
        human_message = HumanMessage(content=message, id=f"human-{uuid.uuid4().hex}")
        run_input = {"messages": [human_message]}
        # Seed history with the user message so rows land in conversation
        # order: every later row appends after it, and the final save dedupes
        # by message id (langgraph preserves the id on the stored message).
        await _save_history(thread_id, username, [_serialize_message(human_message)])

    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)
    finalized: list[dict] = []

    async def put(event: str, data: dict) -> None:
        await queue.put({"event": event, "data": data})

    async def consume_messages(run) -> None:
        counter = 0
        try:
            async for ms in run.messages:
                counter += 1
                ms_id = _field(ms, "message_id", "id", default=f"llm-{counter}")

                async def pipe(projection, event: str, ms_id: str) -> None:
                    try:
                        async for delta in projection:
                            await put(event, {"id": ms_id, "delta": delta})
                    except Exception as exc:
                        await put("error", {"source": "messages", "message": str(exc)})

                async def pipe_reasoning(ms, ms_id: str) -> None:
                    # Thinking is a bracketed lifecycle (start -> delta* ->
                    # end), shaped like tool calls, so each reasoning turn
                    # (one per LLM message) is delimited for consumers.
                    started = False
                    try:
                        async for delta in ms.reasoning:
                            if not started:
                                started = True
                                await put("reasoning_start", {"id": ms_id})
                            await put("reasoning_delta", {"id": ms_id, "delta": delta})
                    except Exception as exc:
                        await put("error", {"source": "messages", "message": str(exc)})
                    if started:
                        await put("reasoning_end", {"id": ms_id})

                # Text + reasoning (thinking) deltas stream concurrently; the
                # finalized message event is emitted only after both finish.
                await asyncio.gather(
                    pipe(ms.text, "message_delta", ms_id),
                    pipe_reasoning(ms, ms_id),
                )
                try:
                    out_attr = ms.output  # awaitable property (async) / property (sync)
                    out = await out_attr() if callable(out_attr) else await out_attr
                except Exception:
                    out = None
                if out is not None:
                    serialized = _serialize_message(out)
                    finalized.append(serialized)
                    # Durability: persist each finalized message as it lands
                    # (deduped by message id), so cancelled or interrupted runs
                    # still leave readable history behind.
                    await _save_history(thread_id, username, [serialized])
                    await put("message", {"id": ms_id, "message": serialized})
        except Exception as exc:
            await put("error", {"source": "messages", "message": str(exc)})

    async def consume_tool_calls(run) -> None:
        try:
            async for tc in run.tool_calls:
                # ToolCallStream: stable fields at start, terminal fields after drain.
                tc_id = _field(tc, "tool_call_id", "id", default=f"call-{uuid.uuid4().hex[:8]}")
                name = _field(tc, "tool_name", "name", default="tool")
                await put(
                    "tool_start",
                    {"id": tc_id, "name": name, "args": _field(tc, "input", "args", default={})},
                )
                try:
                    async for delta in tc.output_deltas:
                        await put("tool_delta", {"id": tc_id, "name": name, "delta": delta})
                except Exception:
                    pass  # no deltas
                error = _field(tc, "error", default=None)
                await put(
                    "tool_end",
                    {
                        "id": tc_id,
                        "name": name,
                        "output": _field(tc, "output", default=None),
                        "is_error": error is not None
                        or bool(_field(tc, "completed", default=True) is False),
                        "error": error,
                    },
                )
        except Exception as exc:
            await put("error", {"source": "tool_calls", "message": str(exc)})

    async def consume_subagents(run) -> None:
        try:
            async for sub in run.subagents:
                name = _field(sub, "name", default="subagent")
                await put("subagent", {"name": name, "status": "started"})
                try:
                    async for ms in sub.messages:
                        async for delta in ms.text:
                            await put("subagent_delta", {"subagent": name, "delta": delta})
                except Exception:
                    pass  # subagent message stream is best-effort
                status = _field(sub, "status", default="completed")
                output = None
                with suppress(Exception):
                    output = await sub.output()
                await put(
                    "subagent",
                    {
                        "name": name,
                        "status": status,
                        "output": output,
                        "error": _field(sub, "error"),
                    },
                )
        except Exception as exc:
            await put("error", {"source": "subagents", "message": str(exc)})

    async def consume_values(run) -> None:
        # Drain state snapshots so the pump never blocks; not emitted for now.
        with suppress(Exception):
            async for _ in run.values:
                pass

    interrupted = {"flag": False}
    stop_event = runs.register(thread_id)
    try:
        # Run with the user's identity in the runtime context: the store-backed
        # filesystem backend and the knowledge base search tool scope per user.
        context = {"user_id": username}
        try:
            run = await agent.astream_events(
                run_input, config=config, version="v3", context=context
            )
        except Exception as exc:
            await _save_history(thread_id, username, finalized)
            metadata = await _record_thread_metadata(
                thread_id, username, message, agent_name, status="failed"
            )
            hub.publish(username, _run_event("failed", thread_id, metadata, agent_name))
            run_manager.publish(active, "error", {"source": "run", "message": str(exc)})
            run_manager.publish(active, "done", {"thread_id": thread_id, "messages": []})
            return

        async def drain() -> None:
            results = await asyncio.gather(
                consume_messages(run),
                consume_tool_calls(run),
                consume_subagents(run),
                consume_values(run),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, GraphInterrupt):
                    interrupted["flag"] = True
                elif isinstance(r, Exception):
                    logger.error("stream consumer failed: %r", r)
            await queue.put(None)

        producer = asyncio.create_task(drain())
        cancelled = False
        try:
            while True:
                get_task = asyncio.create_task(queue.get())
                stop_task = asyncio.create_task(stop_event.wait())
                done_tasks, pending = await asyncio.wait(
                    {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                if stop_task in done_tasks:
                    # POST /threads/{id}/cancel fired: abort the run and let
                    # the stream close with a terminal done event.
                    cancelled = True
                    break
                item = get_task.result()
                if item is None:
                    break
                run_manager.publish(active, item["event"], item["data"])
        finally:
            producer.cancel()

        if cancelled:
            # Persist whatever finalized before the abort (the incremental
            # writes above already captured it; this is the belt-and-braces
            # full save — deduped by message id).
            await _save_history(thread_id, username, finalized)
            metadata = await _record_thread_metadata(
                thread_id, username, message, agent_name, status="cancelled"
            )
            hub.publish(username, _run_event("cancelled", thread_id, metadata, agent_name))
            run_manager.publish(
                active, "done", {"thread_id": thread_id, "messages": [], "cancelled": True}
            )
            return

        try:
            # Final state. With v3 streaming a HITL interrupt does not raise —
            # the run ends normally, so we detect it from the persisted state's
            # pending tasks instead.
            messages: list[dict] = finalized if finalized else []
            try:
                final = await run.output()
                if final:
                    messages = [_serialize_message(m) for m in final.get("messages", [])]
            except GraphInterrupt:
                pass  # older runtimes raise; detected via snapshot below

            snapshot = await agent.aget_state(config)
            interrupts = _collect_interrupts(snapshot)
            usage = _aggregate_usage(messages)
            status = "interrupted" if interrupts else "completed"
            # Persist history + refresh thread metadata before the terminal
            # events, so GET /threads and GET /threads/{id}/messages see the
            # finished run. The rewrite fixes stream-order rows (instant tool
            # calls) back to conversation order.
            await persistence.chat_history.replace_messages(thread_id, username, messages)
            metadata = await _record_thread_metadata(
                thread_id, username, message, agent_name, status=status
            )
            hub.publish(username, _run_event(status, thread_id, metadata, agent_name))
            if interrupts:
                # Run paused for human input: surface the HITL requests.
                run_manager.publish(
                    active, "interrupt", {"thread_id": thread_id, "interrupts": interrupts}
                )
                done_data: dict = {
                    "thread_id": thread_id,
                    "messages": messages,
                    "interrupted": True,
                }
            else:
                done_data = {"thread_id": thread_id, "messages": messages}
            if usage:
                done_data["usage"] = usage
            run_manager.publish(active, "done", done_data)
        except Exception as exc:
            logger.exception("failed to finalize run %s", thread_id)
            await _save_history(thread_id, username, finalized)
            metadata = await _record_thread_metadata(
                thread_id, username, message, agent_name, status="failed"
            )
            hub.publish(username, _run_event("failed", thread_id, metadata, agent_name))
            run_manager.publish(active, "error", {"source": "final", "message": str(exc)})
            run_manager.publish(active, "done", {"thread_id": thread_id, "messages": finalized})
    finally:
        runs.unregister(thread_id)
        run_manager.finish(thread_id)
