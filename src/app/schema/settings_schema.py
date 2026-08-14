"""App settings API models: runtime overrides for .env defaults."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExecuteSettingsIn(BaseModel):
    """execute-tool overrides (None fields keep the current value)."""

    enabled: bool | None = None
    max_timeout: int | None = Field(default=None, ge=1, le=86400)
    inherit_env: bool | None = None


class ConnectionsPolicyIn(BaseModel):
    """Connection resolution policy (None keeps the current value)."""

    fallback_env: bool | None = Field(
        default=None,
        description=(
            "Allow .env credentials when no DB connection of a kind exists. "
            "False (default) = DB connections are mandatory."
        ),
    )


class HitlSettingsIn(BaseModel):
    """Human-in-the-loop gating for the builtin default agent."""

    interrupt_on: dict[str, bool] | None = Field(
        default=None,
        description=(
            'Tool name -> pause for approval, e.g. {"execute": true, '
            '"edit_file": true}. {} / null = HITL off. Named agent configs '
            "keep their own per-config interrupt_on."
        ),
    )


class SettingsIn(BaseModel):
    """Partial update: only the provided keys are written."""

    execute: ExecuteSettingsIn | None = None
    connections: ConnectionsPolicyIn | None = None
    hitl: HitlSettingsIn | None = None


class SettingsOut(BaseModel):
    """Effective settings + where each value comes from."""

    execute: dict = Field(
        default_factory=dict,
        description='e.g. {"enabled": true, "max_timeout": 3600, '
        '"inherit_env": false, "source": "db"|"env"}',
    )
    connections: dict = Field(
        default_factory=dict,
        description='e.g. {"fallback_env": false, "source": "db"|"env"}',
    )
    hitl: dict = Field(
        default_factory=dict,
        description='e.g. {"interrupt_on": {"execute": true}, "source": "db"|"env"}',
    )
