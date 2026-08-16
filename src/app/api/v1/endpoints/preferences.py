"""Per-user preferences routes: GET/PATCH /users/me/preferences.

Preferences live in the durable `user_preferences` store (Postgres JSONB,
in-memory fallback) and persist across sessions, so the frontend keeps the
web search toggle and the chat display toggles (hide thinking / hide tool
calls) server-side instead of localStorage. Chat requests simply omit
`enable_search` to fall back to the stored value (explicit request fields
still win, see `services/searxng.apply_search_preference`); the display
toggles are read from the store when each chat stream starts (see
`services/chat._hidden_prefs`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ....core.config import settings
from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....schema.preferences_schema import PreferencesIn, PreferencesOut

router = APIRouter(tags=["preferences"])

# Preference keys with their defaults: `_CONFIG_DEFAULTS` values come from
# the server config (stored wins); plain display preferences default to False
# (nothing hidden) unless the user stored a True.
_CONFIG_DEFAULTS = {
    "enable_search": lambda: settings.searxng_enabled,
    "hide_reasoning": lambda: False,
    "hide_tool_calls": lambda: False,
}


async def _effective(username: str) -> dict:
    """Stored preferences merged over the server defaults (stored wins)."""
    stored = await persistence.preferences.get_all(username)
    return {key: stored.get(key, default()) for key, default in _CONFIG_DEFAULTS.items()}


@router.get("/users/me/preferences", response_model=PreferencesOut)
async def read_my_preferences(current_user: dict = Depends(get_current_user)):
    """The user's effective preferences: stored value or server default."""
    return await _effective(current_user["username"])


@router.patch("/users/me/preferences", response_model=PreferencesOut)
async def update_my_preferences(
    body: PreferencesIn,
    current_user: dict = Depends(get_current_user),
):
    """Set or clear preference keys; explicit null clears back to the default.

    The response always carries the full effective state so the frontend can
    render the toggle without a follow-up GET.
    """
    username = current_user["username"]
    for key in _CONFIG_DEFAULTS:
        if key in body.model_fields_set:
            await persistence.preferences.set(username, key, getattr(body, key))
    return await _effective(username)
