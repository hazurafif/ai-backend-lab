"""LLM-generated thread titles (prompt template -> upserted thread metadata).

The endpoint `POST /threads/{id}/title` renders the conversation into the
template below, runs it through the thread's own agent model, and stores the
result as the thread title (creating the metadata row when missing). Falls
back to the first user message (truncated) when the model is unavailable or
returns unusable output.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Conversation slice fed to the model: latest messages, each truncated.
MAX_MESSAGES = 12
MAX_MESSAGE_CHARS = 400
MAX_TITLE_CHARS = 80

TITLE_SYSTEM_PROMPT = """\
You are a chat title generator for a conversation archive.
Generate a short, descriptive title for the conversation below.

Rules:
- at most 6 words; lowercase except proper nouns
- no quotes, no trailing punctuation, no "chat about" prefixes
- capture the topic or intent, e.g. "deploy fastapi on fly.io" or
  "fix postgres connection errors"

Reply with the title only.

Conversation:
{conversation}
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are a helpful assistant suggesting follow-up questions for the user.
Read the conversation and propose what the user would naturally ask next.

Rules:
- exactly 3 questions
- short (at most 12 words each), diverse angles (deeper dive, alternatives,
  practical next steps)
- one per line: plain text, no numbering, no bullets, no quotes

Conversation:
{conversation}
"""


def _message_text(m: Any) -> str | None:
    """Plain-text content of a message (skips tool calls / reasoning)."""
    content = getattr(m, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(parts) if parts else None
    return str(content)


def format_conversation(messages: list[Any]) -> str:
    """Render the latest messages as `user:` / `assistant:` lines for the model."""
    lines: list[str] = []
    for m in messages[-MAX_MESSAGES:]:
        role = (
            "user"
            if getattr(m, "type", "") == "human"
            else ("assistant" if getattr(m, "type", "") == "ai" else None)
        )
        if role is None:
            continue
        text = _message_text(m)
        if not text:
            continue
        text = text.strip().replace("\n", " ")
        lines.append(f"{role}: {text[:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines)


def clean_title(raw: str) -> str:
    """Normalize model output into a usable title ("" when unusable)."""
    title = re.sub(r"^[\s\"'.]+|[\s\"'.]+$", "", raw)
    title = re.sub(r"\s+", " ", title)
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rstrip()
    if len(title) < 2:
        return ""
    return title


def _fallback_title(messages: list[Any]) -> str:
    """Deterministic fallback: the first user message, truncated."""
    for m in messages:
        text = _message_text(m)
        if text:
            return text.strip()[:MAX_TITLE_CHARS]
    return "untitled"


async def generate_title(model: str | BaseChatModel, messages: list[Any]) -> str:
    """Generate a title for the conversation; always returns a usable string.

    A plain-string model (env-driven, no DB connection) can't be invoked —
    the deterministic fallback is used instead.
    """
    conversation = format_conversation(messages)
    if not isinstance(model, BaseChatModel):
        return _fallback_title(messages)
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=TITLE_SYSTEM_PROMPT.format(conversation=conversation)),
                HumanMessage(content="Generate the title for this conversation."),
            ]
        )
        content = getattr(response, "content", None)
        raw = (
            content
            if isinstance(content, str)
            else (
                "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                if isinstance(content, list)
                else ""
            )
        )
        return clean_title(raw) or _fallback_title(messages)
    except Exception:
        return _fallback_title(messages)


def _split_question_lines(raw: str) -> list[str]:
    """Parse the model's line-per-question output into clean questions."""
    questions: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
        line = re.sub(r"^[\s\"'.]+|[\s\"'.]+$", "", line)
        line = re.sub(r"\s+", " ", line)
        if 3 <= len(line) <= 100 and not line.endswith(":"):
            questions.append(line)
    return questions[:3]


async def generate_followups(model: str | BaseChatModel, messages: list[Any]) -> list[str]:
    """Suggest up to 3 follow-up questions for the conversation ([] when unavailable)."""
    if not isinstance(model, BaseChatModel):
        return []
    conversation = format_conversation(messages)
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT.format(conversation=conversation)),
                HumanMessage(content="Suggest follow-up questions for this conversation."),
            ]
        )
        content = getattr(response, "content", None)
        raw = (
            content
            if isinstance(content, str)
            else (
                "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                if isinstance(content, list)
                else ""
            )
        )
        return _split_question_lines(raw)
    except Exception:
        return []
