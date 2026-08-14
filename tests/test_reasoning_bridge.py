"""Offline tests for the reasoning_content bridge (ChatOpenAI patch).

ChatOpenAI drops non-standard `reasoning_content` deltas; the bridge lifts
them into reasoning content blocks so the v3 streaming projection surfaces
them. These tests exercise the patched converters directly (no network).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai.chat_models import base as openai_base

from app.services.reasoning_bridge import install

install()  # module import == app behavior (main.py installs at startup)


def _delta(**fields) -> AIMessageChunk:
    return openai_base._convert_delta_to_message_chunk(
        {"role": "assistant", **fields}, AIMessageChunk
    )


def test_streaming_delta_reasoning_lifted():
    chunk = _delta(content="answer text", reasoning_content="let me think...")
    assert chunk.content == [
        {"type": "reasoning", "reasoning": "let me think..."},
        {"type": "text", "text": "answer text"},
    ]


def test_streaming_delta_empty_reasoning_untouched():
    chunk = _delta(content="plain answer")
    assert chunk.content == "plain answer"
    chunk = _delta(content="", reasoning_content="")
    assert chunk.content == ""


def test_streaming_delta_with_tool_calls_keeps_blocks():
    chunk = _delta(content="", reasoning_content="reasoning", tool_calls=None)
    assert chunk.content[0] == {"type": "reasoning", "reasoning": "reasoning"}


def test_non_streaming_message_reasoning_lifted():
    message = openai_base._convert_dict_to_message(
        {"role": "assistant", "content": "final", "reasoning_content": "inner monologue"}
    )
    assert isinstance(message, AIMessage)
    assert message.content == [
        {"type": "reasoning", "reasoning": "inner monologue"},
        {"type": "text", "text": "final"},
    ]


def test_non_streaming_message_untouched():
    message = openai_base._convert_dict_to_message({"role": "assistant", "content": "final"})
    assert message.content == "final"


def test_install_is_idempotent():
    first = openai_base._convert_delta_to_message_chunk
    install()
    assert openai_base._convert_delta_to_message_chunk is first
