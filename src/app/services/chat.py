"""Chat service: runs the agent and normalizes the v3 stream to SSE events.

`agent_stream` is the single entry point used by the `/chat` and `/api/chat`
routers (and the AI SDK bridge). The SSE event contract below is normative —
keep it in sync with the frontend consumers.
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
    thread_id: str, username: str, message: str | None, agent_name: str = "default"
) -> None:
    """Upsert thread metadata in the store: title on first message, `updated_at` on every run."""
    now = now_iso()
    try:
        ns = thread_metadata_ns(username)
        existing = await persistence.store.aget(ns, thread_id)
        value = dict(existing.value) if existing is not None else {}
        if message is not None:
            value.setdefault("title", message[:80])
            value.setdefault("created_at", now)
        value["updated_at"] = now
        value["agent"] = agent_name
        await persistence.store.aput(ns, thread_id, value)
    except Exception:
        logger.exception("failed to record thread metadata for %s", thread_id)


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


async def agent_stream(
    agent: CompiledStateGraph,
    username: str,
    *,
    message: str | None = None,
    thread_id: str | None = None,
    resume: Any = None,
    agent_name: str = "default",
) -> AsyncIterator[str]:
    """Run the agent and forward v3 stream projections as normalized SSE events.

    - `message`: new user message for /chat
    - `resume`: value for `Command(resume=...)` (HITL continuation, /resume)
    - `agent_name`: agent config this thread runs on (recorded in thread metadata)
    """
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    if resume is not None:
        run_input: Any = Command(resume=resume)
    else:
        run_input = {"messages": [HumanMessage(content=message)]}

    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)

    async def put(event: str, data: dict) -> None:
        await queue.put({"event": event, "data": data})

    async def consume_messages(run) -> None:
        counter = 0
        try:
            async for ms in run.messages:
                counter += 1
                ms_id = _field(ms, "message_id", "id", default=f"llm-{counter}")
                try:
                    async for delta in ms.text:
                        await put("message_delta", {"id": ms_id, "delta": delta})
                except Exception as exc:
                    await put("error", {"source": "messages", "message": str(exc)})
                try:
                    out_attr = ms.output  # awaitable property (async) / property (sync)
                    out = await out_attr() if callable(out_attr) else await out_attr
                except Exception:
                    out = None
                if out is not None:
                    await put("message", {"id": ms_id, "message": _serialize_message(out)})
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

    # Run with the user's identity in the runtime context: the store-backed
    # filesystem backend and the knowledge base search tool scope per user.
    context = {"user_id": username}
    try:
        run = await agent.astream_events(run_input, config=config, version="v3", context=context)
    except Exception as exc:
        yield _sse("error", {"source": "run", "message": str(exc)})
        yield _sse("done", {"thread_id": thread_id, "messages": []})
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
    stop_event = runs.register(thread_id)
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
                # POST /threads/{id}/cancel fired: abort the run and close
                # the stream with a terminal done event.
                cancelled = True
                break
            item = get_task.result()
            if item is None:
                break
            yield _sse(item["event"], item["data"])
    finally:
        producer.cancel()
        runs.unregister(thread_id)

    if cancelled:
        # No history/metadata writes for aborted runs (partial state).
        yield _sse("done", {"thread_id": thread_id, "messages": [], "cancelled": True})
        return

    # Final state. With v3 streaming a HITL interrupt does not raise — the
    # run ends normally, so we detect it from the persisted state's pending
    # tasks instead.
    messages: list[dict] = []
    try:
        final = await run.output()
        if final:
            messages = [_serialize_message(m) for m in final.get("messages", [])]
    except GraphInterrupt:
        pass  # older runtimes raise; handled via snapshot below
    except Exception as exc:
        yield _sse("error", {"source": "final", "message": str(exc)})
        yield _sse("done", {"thread_id": thread_id, "messages": messages})
        return

    snapshot = await agent.aget_state(config)
    interrupts = _collect_interrupts(snapshot)
    usage = _aggregate_usage(messages)
    # Persist history + refresh thread metadata before the terminal events, so
    # GET /threads and GET /threads/{id}/messages see the finished run.
    await _save_history(thread_id, username, messages)
    await _record_thread_metadata(thread_id, username, message, agent_name)
    if interrupts:
        # Run paused for human input: surface the HITL requests.
        yield _sse("interrupt", {"thread_id": thread_id, "interrupts": interrupts})
        done_data: dict = {"thread_id": thread_id, "messages": messages, "interrupted": True}
    else:
        done_data = {"thread_id": thread_id, "messages": messages}
    if usage:
        done_data["usage"] = usage
    yield _sse("done", done_data)
