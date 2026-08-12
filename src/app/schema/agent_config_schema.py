"""Agent config models (customizable agent profiles).

An agent config bundles everything that used to be env-global into a named,
persisted profile: the model string, system prompt, a selection of skills and
MCP tool servers. The built-in `default` agent is synthesized from
`DEEPAGENTS_MODEL` / `SYSTEM_PROMPT` env settings and cannot be created or
deleted through the API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .agent_schema import SKILL_NAME_PATTERN

MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$"
"""Light provider:model validation; full resolution happens at agent build time."""

TOOL_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"


class AgentConfigIn(BaseModel):
    """Create/update payload for an agent config."""

    name: str = Field(
        ...,
        pattern=SKILL_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description=(
            "Agent identifier (lowercase alphanumeric + hyphens). Referenced "
            "as `agent` in /chat and /api/chat request bodies. The name "
            "'default' is reserved."
        ),
    )
    description: str | None = Field(default=None, max_length=512)
    model: str = Field(
        ...,
        pattern=MODEL_PATTERN,
        min_length=1,
        max_length=128,
        description=(
            "Provider:model string, e.g. 'openai:gpt-4o-mini', "
            "'anthropic:claude-sonnet-4-5', 'google_genai:gemini-2.5-flash'"
        ),
    )
    system_prompt: str | None = Field(
        default=None, description="Agent instructions; None = fall back to the default prompt"
    )
    skills: list[str] | None = Field(
        default=None,
        description=(
            "Skill names to attach (snapshot-copied from the global skills "
            "store into this agent's namespace). None = inherit the global "
            "skills source; [] = no skills."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "MCP tool server names to attach, plus the special 'web_search'. "
            "None = inherit all configured tools; [] = no MCP tools."
        ),
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    interrupt_on: dict[str, bool] | None = Field(
        default=None,
        description='e.g. {"edit_file": true} -> pause for human approval before edits',
    )
    scope: Literal["user", "global"] = Field(
        default="user",
        description="'global' agents are shared by all users (admin only)",
    )


class AgentConfigOut(BaseModel):
    """Agent config as returned by the API (store value + metadata)."""

    name: str
    description: str | None = None
    model: str
    system_prompt: str | None = None
    skills: list[str] | None = None
    tools: list[str] | None = None
    temperature: float | None = None
    interrupt_on: dict[str, bool] | None = None
    scope: Literal["user", "global"] = "user"
    owner: str = "system"
    builtin: bool = False
    created_at: str | None = None
    updated_at: str | None = None
