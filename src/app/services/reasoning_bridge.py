"""Bridge for `reasoning_content` deltas (DeepSeek, vLLM, OpenRouter, gateways).

`ChatOpenAI` targets the official OpenAI API spec and silently drops the
non-standard `reasoning_content` field that DeepSeek-style providers send on
chat/completions streams (their own docstring says so). This module patches
langchain-openai's two message converters so reasoning deltas are lifted into
standard reasoning content blocks instead:

    {"type": "reasoning", "reasoning": "..."}

The langchain v3 streaming protocol then surfaces them through the
`ms.reasoning` projection, which the chat service forwards as
`reasoning_delta` SSE events / AI SDK `reasoning-*` chunks, and the finalized
message keeps the block in thread history.

The patch is a no-op for providers that never send `reasoning_content`
(official OpenAI chat/completions excluded, Responses API untouched).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_openai import chat_models as _chat_models

_PATCHED = False


def _lift_reasoning(content: Any, reasoning: Any) -> Any:
    """Prepend a reasoning block when the provider sent `reasoning_content`."""
    if not reasoning:
        return content
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    return [{"type": "reasoning", "reasoning": str(reasoning)}, *blocks]


def _wrap_delta_converter(original):
    def converter(
        _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        chunk = original(_dict, default_class)
        if isinstance(chunk, AIMessageChunk):
            chunk.content = _lift_reasoning(chunk.content, _dict.get("reasoning_content"))
        return chunk

    return converter


def _wrap_message_converter(original):
    def converter(_dict: Mapping[str, Any]) -> BaseMessage:
        message = original(_dict)
        if isinstance(message, AIMessage):
            message.content = _lift_reasoning(message.content, _dict.get("reasoning_content"))
        return message

    return converter


def install() -> None:
    """Patch langchain-openai's message converters (idempotent, app startup)."""
    global _PATCHED
    if _PATCHED:
        return
    base = _chat_models.base
    base._convert_delta_to_message_chunk = _wrap_delta_converter(
        base._convert_delta_to_message_chunk
    )
    base._convert_dict_to_message = _wrap_message_converter(base._convert_dict_to_message)
    _PATCHED = True
