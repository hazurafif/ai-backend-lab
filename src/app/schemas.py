from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


# --- Chat / agent API ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the agent")
    thread_id: str | None = Field(
        default=None,
        description="Conversation id. Omit to start a new conversation.",
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
        description="Model id chosen in the UI (informational for now)",
    )
    enable_search: bool | None = Field(
        default=None,
        alias="enableSearch",
        description=(
            "Override the SearXNG web search toggle for this chat "
            "(None = use SEARXNG_ENABLED config)"
        ),
    )


class ResumeRequest(BaseModel):
    """Decision(s) for a human-in-the-loop interrupt.

    Either a single `decision` (one action request) or a full `decisions`
    list (must match the number of `action_requests` in the interrupt
    payload). Decision types: approve | edit | reject | respond.
    """

    decision: dict | None = Field(default=None)
    decisions: list[dict] | None = Field(default=None)
