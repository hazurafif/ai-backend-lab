"""Per-user model allowlists (admin-managed).

Admins restrict which models a user may use by setting an allowlist of model
ids (as shown by `GET /connections/models`, e.g. "openai:deepseek-v4-flash"
or "gemini-2.5-pro"). Semantics:

- **No row** (never set, or cleared) → unrestricted: every model is allowed.
- **Row present** → only the listed model ids are allowed (`[]` = none).

Persistence mirrors the other resource services: LangGraph store under
`("user", "allowed_models", <username>)`, key `models` = `{"models": [...]}`,
Postgres-backed in production, in-memory in dev.

Enforcement happens at agent-config create/update (user scope) and when a
chat run resolves an agent (so the builtin default agent and configs created
before a restriction also honor it). Global (admin-made) agent configs are
never restricted — admins manage them.
"""

from __future__ import annotations

from langgraph.store.base import BaseStore

from ..core.constants import user_allowed_models_ns

_KEY = "models"


async def get_allowed_models(store: BaseStore, username: str) -> list[str]:
    """The user's allowlist (empty when unrestricted)."""
    item = await store.aget(user_allowed_models_ns(username), _KEY)
    if item is None:
        return []
    value = item.value or {}
    models = value.get("models") or []
    return [m for m in models if isinstance(m, str)]


async def is_restricted(store: BaseStore, username: str) -> bool:
    """True when the admin set an allowlist for this user (even an empty one)."""
    return await store.aget(user_allowed_models_ns(username), _KEY) is not None


async def is_model_allowed(store: BaseStore, username: str, model: str) -> bool:
    """Whether `model` may be used: unrestricted, or listed in the allowlist.

    A set allowlist always restricts (an empty one allows nothing).
    """
    if not model:
        return False
    if not await is_restricted(store, username):
        return True
    return model in await get_allowed_models(store, username)


async def set_allowed_models(store: BaseStore, username: str, models: list[str]) -> list[str]:
    """Replace the allowlist (empty list = allow nothing). Returns the stored list."""
    cleaned = [m for m in models if isinstance(m, str)]
    await store.aput(user_allowed_models_ns(username), _KEY, {"models": cleaned})
    return cleaned


async def clear_allowed_models(store: BaseStore, username: str) -> bool:
    """Remove the allowlist (back to unrestricted); False when none was set."""
    ns = user_allowed_models_ns(username)
    if await store.aget(ns, _KEY) is None:
        return False
    await store.adelete(ns, _KEY)
    return True
