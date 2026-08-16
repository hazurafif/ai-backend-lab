"""Global model allowlist for the `user` role (admin-managed).

A single allowlist of model ids (as shown by `GET /connections/models`, e.g.
"openai:deepseek-v4-flash" or "gemini-2.5-pro") applies to every non-admin
user — there is no per-user configuration. Semantics:

- **Key absent** (never set, or cleared) → unrestricted: every model is
  allowed.
- **Key present** → only the listed model ids are allowed for `user`-role
  accounts (`[]` = none). Admins are never restricted (they manage the list).

Persistence lives in the `app_settings` store (Postgres `app_settings`
table / in-memory fallback) under the `model_allowlist` key, the same
admin-managed store as the execute/HITL settings.

Enforcement happens at agent-config create/update (user-scoped configs) and
when a chat run resolves an agent (so the builtin default agent and configs
created before a restriction also honor it). Guest chats (no auth) count as
`user` role. Global (admin-made) agent configs are never restricted.
"""

from __future__ import annotations

from typing import Any

from ..core.database import persistence

_KEY = "model_allowlist"


async def get_allowed_models() -> list[str]:
    """The configured allowlist (empty when unrestricted)."""
    value = await persistence.settings.get(_KEY)
    models = (value or {}).get("models") or []
    return [m for m in models if isinstance(m, str)]


async def is_restricted() -> bool:
    """True when an allowlist is configured (even an empty one)."""
    return await persistence.settings.get(_KEY) is not None


async def is_model_allowed(model: str) -> bool:
    """Whether `model` may be used by a `user`-role account.

    Unrestricted -> any model; restricted -> only the listed ids
    (an empty list allows nothing).
    """
    if not model:
        return False
    if not await is_restricted():
        return True
    return model in await get_allowed_models()


async def set_allowed_models(models: list[str]) -> list[str]:
    """Replace the allowlist (empty list = allow nothing). Returns the stored list."""
    cleaned = [m for m in models if isinstance(m, str)]
    await persistence.settings.set(_KEY, {"models": cleaned})
    return cleaned


async def clear_allowed_models() -> bool:
    """Remove the allowlist (back to unrestricted); False when none was set."""
    if not await is_restricted():
        return False
    await persistence.settings.delete(_KEY)
    return True


def role_allows_all(role: str | None) -> bool:
    """Admins bypass the allowlist; users (and guests) are restricted by it."""
    return role == "admin"


async def effective_for_role(role: str | None) -> dict[str, Any]:
    """The restriction visible to one account: (restricted, models).

    Admins always see `restricted=False` (they can use every model).
    """
    return {
        "restricted": not role_allows_all(role) and await is_restricted(),
        "models": await get_allowed_models(),
    }
