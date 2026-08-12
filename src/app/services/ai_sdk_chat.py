"""AI SDK data-stream protocol bridge.

The frontend (Vercel AI SDK ``useChat``) talks to backends over the
"data stream protocol": an SSE stream whose ``data:`` lines carry typed JSON
chunks (``start``, ``text-start``/``text-delta``/``text-end``, ``custom``,
``error``, ``finish``), terminated by ``data: [DONE]``.

This module bridges that protocol to the agent's normalized SSE events
(``agent_stream`` in ``app/services/chat.py``) and extracts the user prompt from the AI
SDK request body. Endpoint: ``POST /api/chat``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


def extract_user_message(messages: list[dict[str, Any]]) -> str:
    """Return the text of the last user message in an AI SDK request body.

    Handles both the parts format (``[{"type": "text", "text": ...}]``) and
    legacy plain-string ``content``.
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        parts = msg.get("parts")
        if isinstance(parts, list):
            texts = [
                p["text"]
                for p in parts
                if isinstance(p, dict)
                and p.get("type") == "text"
                and isinstance(p.get("text"), str)
                and p["text"].strip()
            ]
            if texts:
                return "\n".join(texts)
    return ""


def _chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _parse_event(sse_chunk: str) -> tuple[str, dict[str, Any]]:
    """Parse one SSE chunk (`event: <name>\ndata: <json>`) into (name, data)."""
    ev, _, rest = sse_chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


async def sdk_stream(
    events: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Translate agent SSE events into AI SDK data-stream chunks.

    ``events`` is the SSE chunk generator produced by ``agent_stream``.
    Yields an SSE stream of ``data:`` JSON chunks that ``useChat``
    understands: ``start`` -> ``text-*`` + native ``tool-input-*``/
    ``tool-output-*`` chunks (typed ``tool-<name>`` UI parts) + optional
    ``custom`` subagent chunks -> ``finish``, then ``[DONE]``.
    """
    response_id = f"resp-{uuid.uuid4().hex[:12]}"
    started_text: set[str] = set()
    emitted_error = False
    finished = False
    interrupt_seen = False

    yield _chunk({"type": "start", "messageId": response_id})

    async for sse_chunk in events:
        name, data = _parse_event(sse_chunk)
        if name == "message_delta":
            ms_id = data.get("id") or response_id
            if ms_id not in started_text:
                started_text.add(ms_id)
                yield _chunk({"type": "text-start", "id": ms_id})
            yield _chunk({"type": "text-delta", "id": ms_id, "delta": data.get("delta", "")})
        elif name == "message":
            ms_id = data.get("id") or response_id
            if ms_id in started_text:
                yield _chunk({"type": "text-end", "id": ms_id})
        elif name == "tool_start":
            # Native AI SDK tool chunks -> typed `tool-<name>` UI parts.
            tool_id = data.get("id")
            tool_name = data.get("name")
            args = data.get("args") or {}
            if tool_id is not None and tool_name is not None:
                yield _chunk(
                    {"type": "tool-input-start", "toolCallId": tool_id, "toolName": tool_name}
                )
                yield _chunk(
                    {
                        "type": "tool-input-delta",
                        "toolCallId": tool_id,
                        "inputTextDelta": json.dumps(args, ensure_ascii=False, default=str),
                    }
                )
                yield _chunk(
                    {
                        "type": "tool-input-available",
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "input": args,
                    }
                )
        elif name == "tool_end":
            tool_id = data.get("id")
            if tool_id is not None:
                if data.get("is_error"):
                    yield _chunk(
                        {
                            "type": "tool-output-error",
                            "toolCallId": tool_id,
                            "errorText": str(
                                data.get("error") or data.get("output") or "Tool failed"
                            ),
                        }
                    )
                else:
                    yield _chunk(
                        {
                            "type": "tool-output-available",
                            "toolCallId": tool_id,
                            "output": data.get("output"),
                        }
                    )
        elif name == "subagent":
            # providerMetadata must be keyed by provider name (the AI SDK's
            # providerMetadataSchema is Record<provider, Record<key, value>>)
            # — flat {name, status, error} fails stream validation client-side.
            yield _chunk(
                {
                    "type": "custom",
                    "kind": "app.subagent",
                    "providerMetadata": {
                        "app": {
                            "name": data.get("name"),
                            "status": data.get("status"),
                            "error": data.get("error"),
                        },
                    },
                }
            )
        elif name == "interrupt":
            # Human-in-the-loop pause: surface the approval request as a
            # custom chunk; the frontend shows an approval UI and resumes by
            # posting to /api/chat again with `id` (the thread) + `decision`.
            # providerMetadata must be keyed by provider name (the AI SDK's
            # providerMetadataSchema is Record<provider, Record<key, value>>)
            # — flat extra fields fail the strict uiMessageChunkSchema and
            # kill the whole stream client-side.
            yield _chunk(
                {
                    "type": "custom",
                    "kind": "app.interrupt",
                    "providerMetadata": {
                        "app": {
                            "threadId": data.get("thread_id"),
                            "interrupts": data.get("interrupts"),
                        },
                    },
                }
            )
            interrupt_seen = True
        elif name == "error":
            yield _chunk(
                {
                    "type": "error",
                    "errorText": (
                        f"[{data.get('source', 'agent')}] {data.get('message', 'unknown error')}"
                    ),
                }
            )
            emitted_error = True
            break
        elif name == "done":
            if data.get("cancelled"):
                yield _chunk({"type": "finish", "finishReason": "stop"})
                finished = True
            elif data.get("interrupted"):
                if not interrupt_seen:
                    yield _chunk(
                        {
                            "type": "error",
                            "errorText": "Agent run was interrupted (human-in-the-loop).",
                        }
                    )
                    emitted_error = True
                else:
                    # Interrupt already surfaced as an app.interrupt custom
                    # chunk; close the stream so the UI can render the
                    # approval flow.
                    yield _chunk({"type": "finish", "finishReason": "other"})
                    finished = True
            elif not emitted_error:
                yield _chunk({"type": "finish", "finishReason": "stop"})
                finished = True
            break

    if not emitted_error and not finished:
        yield _chunk({"type": "finish", "finishReason": "stop"})
    yield "data: [DONE]\n\n"
