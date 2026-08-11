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

from ..core.constants import SSE_HEADERS
from ..db import persistence
from ..utils import now_iso

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


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\ndata: {json.dumps(_jsonable(data), ensure_ascii=False, default=str)}\n\n"
    )


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
) -> AsyncIterator[str]:
    """Run the agent and forward v3 stream projections as normalized SSE events.

    - `message`: new user message for /chat
    - `resume`: value for `Command(resume=...)` (HITL continuation, /resume)
    """
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    if message is not None:
        # Record thread metadata in the store so GET /threads can list conversations.
        now = now_iso()
        try:
            await persistence.store.aput(
                ("threads", username),
                thread_id,
                {"title": message[:80], "created_at": now, "updated_at": now},
            )
        except Exception:
            logger.exception("failed to record thread metadata")

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

    # Start the run.
    try:
        run = await agent.astream_events(run_input, config=config, version="v3")
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
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item["event"], item["data"])
    finally:
        producer.cancel()

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
    if interrupts:
        # Run paused for human input: surface the HITL requests.
        yield _sse("interrupt", {"thread_id": thread_id, "interrupts": interrupts})
        yield _sse("done", {"thread_id": thread_id, "messages": messages, "interrupted": True})
    else:
        yield _sse("done", {"thread_id": thread_id, "messages": messages})
