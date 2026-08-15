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
