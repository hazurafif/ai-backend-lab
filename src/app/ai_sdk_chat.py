"""AI SDK data-stream protocol bridge.

The frontend (Vercel AI SDK ``useChat``) talks to backends over the
"data stream protocol": an SSE stream whose ``data:`` lines carry typed JSON
chunks (``start``, ``text-start``/``text-delta``/``text-end``, ``custom``,
``error``, ``finish``), terminated by ``data: [DONE]``.

This module bridges that protocol to the agent's normalized SSE events
(``_agent_stream`` in ``main.py``) and extracts the user prompt from the AI
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

    ``events`` is the SSE chunk generator produced by ``_agent_stream``.
    Yields an SSE stream of ``data:`` JSON chunks that ``useChat``
    understands: ``start`` -> ``text-*`` (+ optional ``custom`` tool/subagent
    chunks) -> ``finish``, then ``[DONE]``.

    Tool calls and subagent activity are forwarded as ``custom`` chunks so the
    UI can render them later; text deltas carry the answer.
    """
    response_id = f"resp-{uuid.uuid4().hex[:12]}"
    started_text: set[str] = set()
    emitted_error = False
    finished = False

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
            yield _chunk(
                {
                    "type": "custom",
                    "kind": "tool-start",
                    "providerMetadata": {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "args": data.get("args"),
                    },
                }
            )
        elif name == "tool_end":
            yield _chunk(
                {
                    "type": "custom",
                    "kind": "tool-end",
                    "providerMetadata": {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "is_error": data.get("is_error"),
                    },
                }
            )
        elif name == "subagent":
            yield _chunk(
                {
                    "type": "custom",
                    "kind": "subagent",
                    "providerMetadata": {
                        "name": data.get("name"),
                        "status": data.get("status"),
                    },
                }
            )
        elif name == "interrupt":
            # Human-in-the-loop pause: surface as an error so useChat stops
            # and the UI can show the toast. Resume flow (POST
            # /threads/{id}/resume) is not wired into this protocol yet.
            yield _chunk(
                {
                    "type": "error",
                    "errorText": (
                        "Agent paused for human approval (resume flow not wired to this UI yet)."
                    ),
                }
            )
            emitted_error = True
            break
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
            if data.get("interrupted") and not emitted_error:
                yield _chunk(
                    {
                        "type": "error",
                        "errorText": (
                            "Agent run was interrupted (human-in-the-loop). "
                            "Resume is not wired to this UI yet."
                        ),
                    }
                )
                emitted_error = True
            elif not emitted_error:
                yield _chunk({"type": "finish", "finishReason": "stop"})
                finished = True
            break

    if not emitted_error and not finished:
        yield _chunk({"type": "finish", "finishReason": "stop"})
    yield "data: [DONE]\n\n"
