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
    status: str | None = Field(
        default=None,
        description=(
            "Run status: running | completed | interrupted | cancelled | failed "
            "(null for threads created before status tracking)"
        ),
    )


class NotificationOut(BaseModel):
    """A run lifecycle event (notifications stream + recent list).

    `type` is one of run_started | run_completed | run_interrupted |
    run_cancelled | run_failed; `seq` is a per-user monotonic counter used as
    the `since` cursor for reconnecting the notifications stream.
    """

    event_id: str
    seq: int
    type: str
    thread_id: str
    title: str | None = None
    agent: str | None = None
    status: str | None = None
    at: str | None = None


class FollowUpIn(BaseModel):
    """Body for the post-run follow-up endpoint (all fields optional)."""

    force: bool = Field(
        default=False,
        description="Regenerate the title even when an intentional one exists.",
    )


class FollowUpOut(BaseModel):
    """Result of a thread follow-up call."""

    thread_id: str
    title: str
    generated: bool = Field(
        description="True when the LLM generated a new title, False when the existing one was kept."
    )
    followups: list[str] = Field(
        default_factory=list,
        description=(
            "Suggested follow-up questions (up to 3) the user can click to continue "
            "the conversation. Empty when generation is unavailable."
        ),
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


class SessionMessagesOut(BaseModel):
    """Message-level size of a thread (session)."""

    count: int = Field(description="Number of stored messages (all roles)")
    characters: int = Field(description="Total content length in characters")


class SessionUsageOut(BaseModel):
    """Cumulative token usage of a thread, from AIMessage usage_metadata.

    Fields follow the langchain `usage_metadata` standard so the numbers are
    comparable across providers. Note that `input_tokens` is summed per run
    and therefore counts history multiple times (billed input, not unique
    tokens); `output_tokens` is additive and accurate.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    runs: int = Field(description="Runs whose usage was reported")


class SessionContextOut(BaseModel):
    """Current context state of a thread against the model's context window.

    `current_input_tokens` is the input token count of the most recent run —
    the context the model actually sees (history included). The window comes
    from a curated model table (best effort); unknown models report null.
    """

    current_input_tokens: int
    context_window: int | None = None
    utilization: float | None = Field(
        default=None,
        description="current_input_tokens / context_window (0..1), null when the window is unknown",
    )
    remaining_tokens: int | None = Field(
        default=None, description="context_window - current_input_tokens, null when unknown"
    )


class ThreadUsageOut(BaseModel):
    """Context + usage report for one thread (the "current session" view)."""

    thread_id: str
    agent: str | None = Field(default=None, description="Agent config the thread runs on")
    model: str | None = Field(default=None, description="Provider:model string of the agent config")
    messages: SessionMessagesOut
    usage: SessionUsageOut | None = Field(
        default=None, description="Cumulative token usage; null when the provider reports none"
    )
    context: SessionContextOut | None = Field(
        default=None,
        description="Current context vs the model window; null before the first run",
    )
    active_run: bool = Field(
        default=False, description="True while a run is in progress on this thread"
    )


class ResumeRequest(BaseModel):
    """Decision(s) for a human-in-the-loop interrupt.

    Either a single `decision` (one action request) or a full `decisions`
    list (must match the number of `action_requests` in the interrupt
    payload). Decision types: approve | edit | reject | respond.
    """

    decision: dict | None = Field(default=None)
    decisions: list[dict] | None = Field(default=None)
