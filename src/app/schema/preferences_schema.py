"""Per-user preferences API models (web search toggle, ...)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreferencesIn(BaseModel):
    """Body for PATCH /users/me/preferences (all fields optional).

    Omitted keys are left untouched; an explicit `null` clears the stored
    value, reverting that key to its server default.
    """

    enable_search: bool | None = Field(
        default=None,
        description=(
            "Persisted web search toggle. Set true/false to override the "
            "SEARXNG_ENABLED server default for this user; null clears the "
            "preference (server default applies again)."
        ),
    )
    hide_reasoning: bool | None = Field(
        default=None,
        description=(
            "Persisted display toggle: hide the model's thinking (reasoning) "
            "from the chat stream. Set true/false to override the default "
            "(show); null clears the preference (show again)."
        ),
    )
    hide_tool_calls: bool | None = Field(
        default=None,
        description=(
            "Persisted display toggle: hide tool-call activity from the chat "
            "stream. Set true/false to override the default (show); null "
            "clears the preference (show again)."
        ),
    )


class PreferencesOut(BaseModel):
    """Effective preference values of the current user.

    `enable_search` is what the agent uses when a chat request omits the
    field: the stored preference when set, else the SEARXNG_ENABLED server
    default.
    """

    enable_search: bool | None = Field(
        default=None,
        description=(
            "Effective web search toggle: the stored per-user preference, "
            "else the SEARXNG_ENABLED server default"
        ),
    )
    hide_reasoning: bool = Field(
        default=False,
        description=(
            "Effective display preference: True hides the model's thinking "
            "(reasoning_start/delta/end events + reasoning blocks) from the "
            "chat stream; False (default) shows it."
        ),
    )
    hide_tool_calls: bool = Field(
        default=False,
        description=(
            "Effective display preference: True hides tool-call activity "
            "(tool_start/delta/end events + tool messages) from the chat "
            "stream; False (default) shows it."
        ),
    )
