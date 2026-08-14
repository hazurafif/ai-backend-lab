"""Runtime app settings: DB-backed overrides for .env defaults.

Settings are admin-managed key-value rows in `core/database.AppSettingsStore`
(Postgres `app_settings` table, in-memory in dev). Two keys exist:

- `execute`   -> {"enabled": bool, "max_timeout": int, "inherit_env": bool}
- `connections` -> {"fallback_env": bool}

DB values win over .env; when a key has no row the corresponding env/default
from `core/config` applies. The whole table is cached process-wide (refreshed
at startup and after every mutation) and read synchronously by agent build
code, mirroring `services/connections`.
"""

from __future__ import annotations

from typing import Any

from ..core.config import settings

_cache: dict[str, dict] = {}


async def refresh_app_settings() -> None:
    """Reload all settings from the store (startup + after mutations)."""
    from ..core.database import persistence

    _cache.clear()
    for row in await persistence.settings.list():
        _cache[row["key"]] = row["value"]


async def update_app_setting(key: str, value: dict[str, Any]) -> None:
    """Persist a setting key and refresh the cache."""
    from ..core.database import persistence

    await persistence.settings.set(key, value)
    _cache[key] = dict(value)


def _row(key: str) -> dict | None:
    return _cache.get(key)


def get_setting(key: str) -> dict | None:
    """The cached stored value of a setting key, or None when unset."""
    return _row(key)


# ---------------------------------------------------------------------------
# execute tool
# ---------------------------------------------------------------------------


def execute_enabled() -> bool:
    """Effective execute-tool toggle: DB `execute.enabled` else EXECUTE_ENABLED."""
    row = _row("execute")
    if row is not None and "enabled" in row:
        return bool(row["enabled"])
    return settings.execute_enabled


def execute_max_timeout() -> int:
    """Effective per-command timeout cap: DB else EXECUTE_MAX_TIMEOUT."""
    row = _row("execute")
    if row is not None and row.get("max_timeout") is not None:
        return int(row["max_timeout"])
    return settings.execute_max_timeout


def execute_inherit_env() -> bool:
    """Effective env inheritance for the shell backend: DB else EXECUTE_INHERIT_ENV."""
    row = _row("execute")
    if row is not None and "inherit_env" in row:
        return bool(row["inherit_env"])
    return settings.execute_inherit_env


# ---------------------------------------------------------------------------
# connection policy
# ---------------------------------------------------------------------------


def connection_fallback_env() -> bool:
    """Whether .env credentials are used when no DB connection of a kind exists.

    Default (false) = the DB connection is mandatory: the agent LLM model and
    KB embeddings fail loudly instead of silently reading .env keys. Env
    fallback stays opt-in (CONNECTION_FALLBACK_ENV=true or PUT /settings).
    """
    row = _row("connections")
    if row is not None and "fallback_env" in row:
        return bool(row["fallback_env"])
    return settings.connection_fallback_env
