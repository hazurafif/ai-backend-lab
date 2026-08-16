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

# Reasoning-effort levels for thinking models (OpenAI ReasoningEffort set,
# also used by Anthropic's effort parameter). "xhigh" = extra high.
THINKING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
THINKING_LEVEL = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


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
    connection: str | None = Field(
        default=None,
        description=(
            "Saved llm connection (see /connections) serving this agent's "
            "model — its base_url + api_token route the completions. None = "
            "the default llm connection. Combined with the aggregated model "
            "list (GET /connections/models) this lets each agent pick a model "
            "from any configured source."
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
    thinking: THINKING_LEVEL | None = Field(
        default=None,
        description=(
            "Reasoning effort / thinking level for the model: none, minimal, low, "
            "medium, high, xhigh (extra high), max. Passed as `reasoning_effort` "
            "to OpenAI-compatible endpoints (incl. stored connections). "
            "None = provider default."
        ),
    )
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
    model: str | None = Field(
        default=None,
        description=(
            "Provider:model string. None on the builtin default agent when "
            "no default llm connection is saved — its extra.model is used "
            "then (there is no env fallback)."
        ),
    )
    connection: str | None = Field(
        default=None,
        description=(
            "Saved llm connection serving this agent's model; None = the default llm connection."
        ),
    )
    system_prompt: str | None = None
    skills: list[str] | None = None
    tools: list[str] | None = None
    temperature: float | None = None
    thinking: THINKING_LEVEL | None = None
    interrupt_on: dict[str, bool] | None = None
    scope: Literal["user", "global"] = "user"
    owner: str = "system"
    builtin: bool = False
    created_at: str | None = None
    updated_at: str | None = None
