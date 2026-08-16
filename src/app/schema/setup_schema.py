"""Setup / onboarding API models.

Connections (the completions API: base URL + key + model) are **admin-managed
and global** — regular users never touch credentials. The setup flow covers
what IS per-user: display/search preferences and MCP tool servers, plus a
read-only status view of the admin-configured model/connection so the
frontend can render a startup screen.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .agent_config_schema import AgentConfigOut
from .agent_schema import ToolServerIn, ToolServerOut
from .connection_schema import ConnectionOut
from .preferences_schema import PreferencesIn, PreferencesOut


class OnboardingIn(BaseModel):
    """Startup payload: the user's own preferences + MCP tool servers.

    The completions API itself is admin-managed (POST /connections) — this
    flow only configures per-user data.
    """

    preferences: PreferencesIn | None = Field(
        default=None,
        description="Per-user preferences (web search, hide thinking / tool calls), saved to the DB",
    )
    mcp_servers: list[ToolServerIn] = Field(
        default_factory=list,
        description="MCP tool servers the user brings (per-user, upserted)",
    )


class SetupOut(BaseModel):
    """The current setup state (what the frontend renders at startup)."""

    completed: Literal[True] | bool = Field(
        ...,
        description="True once a default llm connection (admin-managed) is configured",
    )
    llm_connection: ConnectionOut | None = Field(
        default=None,
        description="The admin-managed default llm connection (token masked), or None",
    )
    model: str | None = Field(
        default=None,
        description="Effective model of the builtin default agent (the llm connection's model)",
    )
    agent: AgentConfigOut | None = Field(
        default=None,
        description="The builtin default agent config as the user sees it",
    )
    mcp_servers: list[ToolServerOut] = Field(
        default_factory=list,
        description="The user's configured MCP tool servers (per-user)",
    )
    preferences: PreferencesOut = Field(
        ..., description="Effective per-user preferences (stored values, else defaults)"
    )
