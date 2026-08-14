"""API connection models: LLM provider credentials (base URL + API key).

A connection is a named pair of `base_url` + `api_key` for an
OpenAI-compatible endpoint (e.g. `https://api.openai.com/v1`). Connections
are persisted in the durable store and used to build the agent's chat model,
so no `.env` keys are needed at runtime. The API key is write-only: it is
never returned by the API (responses carry `has_api_key` instead).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .agent_schema import SKILL_NAME_PATTERN

CONNECTION_NAME_PATTERN = SKILL_NAME_PATTERN
"""Connection identifiers: lowercase alphanumeric + hyphens (like skills)."""


class ConnectionIn(BaseModel):
    """Create payload for an API connection."""

    name: str = Field(
        ...,
        pattern=CONNECTION_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description=(
            "Connection name, referenced by agent configs (`connection` field). "
            "The name 'default' is used by the builtin default agent."
        ),
    )
    base_url: str = Field(
        ...,
        min_length=8,
        max_length=512,
        description=("Base URL of the OpenAI-compatible endpoint, e.g. https://api.openai.com/v1"),
    )
    api_key: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="API key for the endpoint (stored; never returned by the API)",
    )


class ConnectionUpdate(BaseModel):
    """Update payload: omitted fields keep their stored values.

    `api_key` omitted (or null) keeps the existing key, so rotating the base
    URL alone does not require re-sending the secret.
    """

    base_url: str | None = Field(
        default=None,
        min_length=8,
        max_length=512,
        description="New base URL, or keep the stored one",
    )
    api_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description="New API key, or keep the stored one",
    )


class ConnectionOut(BaseModel):
    """Connection as returned by the API (never includes the API key)."""

    name: str
    base_url: str
    has_api_key: bool = True
    created_at: str
    updated_at: str
