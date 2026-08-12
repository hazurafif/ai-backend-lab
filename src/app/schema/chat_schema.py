"""Chat / agent API models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the agent")
    thread_id: str | None = Field(
        default=None,
        description="Conversation id. Omit to start a new conversation.",
    )
    agent: str | None = Field(
        default=None,
        description=(
            "Agent config name (customizable profile: model + system prompt + "
            "skills + tools). Omit for the built-in 'default' agent."
        ),
    )
    enable_search: bool | None = Field(
        default=None,
        description=(
            "Override the SearXNG web search toggle for this request "
            "(None = use SEARXNG_ENABLED config)"
        ),
    )


class ThreadOut(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str | None = None
    agent: str | None = Field(
        default=None,
        description="Agent config the thread runs on ('default' when not set)",
    )
    share_token: str | None = Field(
        default=None,
        description="Public share token when the thread has been shared",
    )


class ShareOut(BaseModel):
    """Share-link payload: the token plus a URL to view the thread publicly."""

    share_token: str
    url: str


class SharedChatOut(BaseModel):
    """Public view of a shared thread (served without authentication)."""

    thread_id: str
    title: str | None = None
    username: str
    created_at: str | None = None
    messages: list[dict[str, Any]]


class AiSdkChatRequest(BaseModel):
    """Request body of the AI SDK data-stream endpoint (frontend useChat)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(
        default=None,
        description="Chat id from the frontend; reused as the agent thread_id",
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="AI SDK UIMessages; the last user message becomes the prompt",
    )
    selected_chat_model: str | None = Field(
        default=None,
        alias="selectedChatModel",
        description=(
            "Model id chosen in the UI; superseded by `agent` (the agent config "
            "pins the model). Kept for backward compatibility."
        ),
    )
    agent: str | None = Field(
        default=None,
        description=(
            "Agent config name (customizable profile: model + system prompt + "
            "skills + tools). Omit for the built-in 'default' agent."
        ),
    )
    enable_search: bool | None = Field(
        default=None,
        alias="enableSearch",
        description=(
            "Override the SearXNG web search toggle for this chat "
            "(None = use SEARXNG_ENABLED config)"
        ),
    )
    decision: dict | None = Field(
        default=None,
        description=(
            "HITL resume: single decision (approve | edit | reject | respond) "
            "for a paused run; requires `id` = the paused thread"
        ),
    )
    decisions: list[dict] | None = Field(
        default=None,
        description="HITL resume: one decision per action_request of the interrupt",
    )


class ThreadUpdate(BaseModel):
    """Rename payload for PATCH /threads/{id}."""

    title: str = Field(min_length=1, max_length=120)


class ResumeRequest(BaseModel):
    """Decision(s) for a human-in-the-loop interrupt.

    Either a single `decision` (one action request) or a full `decisions`
    list (must match the number of `action_requests` in the interrupt
    payload). Decision types: approve | edit | reject | respond.
    """

    decision: dict | None = Field(default=None)
    decisions: list[dict] | None = Field(default=None)
