"""Connection models: provider credentials (base URL + API token) via API instead of .env.

A connection describes how to reach an external service: an OpenAI-compatible
LLM provider (`llm`), the embeddings provider (`embeddings`), MCP servers
(`mcp`), Weaviate (`weaviate`) or SearXNG (`searxng`). The agent resolves the
default connection per kind at startup (and after every CRUD mutation), so
.env values are only a fallback when no connection is saved.

`api_token` is write-only: outputs mask it (`sk-…wxyz`) and PUT keeps the
stored token when the field is omitted.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CONNECTION_NAME_PATTERN = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$"
ConnectionKind = Literal["llm", "embeddings", "mcp", "weaviate", "searxng"]


class ConnectionIn(BaseModel):
    """Create payload for a provider connection."""

    name: str = Field(
        ...,
        pattern=CONNECTION_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description="Connection identifier, e.g. 'openai' or 'my-vllm'",
    )
    kind: ConnectionKind = "llm"
    base_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Provider base URL, e.g. https://api.openai.com/v1",
    )
    api_token: str | None = Field(
        default=None,
        max_length=4096,
        description="API token / secret. Never returned by the API (masked on output).",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific options (e.g. model, headers, timeout).",
    )
    is_default: bool = Field(
        default=False,
        description="Resolved by consumers of this kind when multiple exist; one default per kind.",
    )
    enabled: bool = Field(
        default=True,
        description="False disables the connection: it is skipped by default resolution, model discovery and agent binding (the row stays editable).",
    )


class ConnectionOut(BaseModel):
    """Connection as returned by the API (api_token masked)."""

    id: str
    name: str
    kind: ConnectionKind = "llm"
    base_url: str | None = None
    api_token: str | None = Field(
        default=None, description="Masked token (first 4 + last 4 chars); full value is write-only."
    )
    has_token: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    enabled: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class ModelOut(BaseModel):
    """One model id exposed by a connection's OpenAI-compatible /models endpoint."""

    id: str
    created: int | None = None
    owned_by: str | None = None


class ModelsSourceOut(BaseModel):
    """A connection's model list (best-effort: `error` when its endpoint failed)."""

    connection: str
    base_url: str | None = None
    is_default: bool = False
    models: list[ModelOut] = Field(default_factory=list)
    error: str | None = None
