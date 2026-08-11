from typing import Any, Literal

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


# --- Agent resources (skills, MCP tool servers) ---

SKILL_NAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class SkillIn(BaseModel):
    """Create/update payload for an agent skill (stored as SKILL.md)."""

    name: str = Field(
        ...,
        pattern=SKILL_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description=(
            "Skill identifier: lowercase alphanumeric and hyphens (per the Agent Skills spec)"
        ),
    )
    description: str = Field(..., min_length=1, max_length=1024)
    content: str = Field(..., min_length=1, description="Markdown instructions body")


class SkillOut(BaseModel):
    name: str
    description: str = ""
    content: str
    path: str = ""


class ToolServerIn(BaseModel):
    """Create/update payload for an MCP tool server (gofastmcp)."""

    name: str = Field(
        ...,
        pattern=SKILL_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description="Server name, referenced in tool names and /health",
    )
    transport: Literal["streamable_http", "stdio"] = "streamable_http"
    url: str | None = Field(default=None, description="Required for streamable_http")
    command: str | None = Field(default=None, description="Required for stdio")
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    def config(self) -> dict:
        """Shape expected by MultiServerMCPClient (mcp_servers.json format)."""
        cfg: dict = {"transport": self.transport}
        if self.transport == "streamable_http":
            cfg["url"] = self.url
            if self.headers:
                cfg["headers"] = self.headers
        else:
            cfg["command"] = self.command
            if self.args:
                cfg["args"] = self.args
            if self.env:
                cfg["env"] = self.env
        return cfg


class ToolServerOut(BaseModel):
    name: str
    transport: str = ""
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
