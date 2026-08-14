"""Settings routes: runtime overrides for .env defaults (admin-only).

GET/PUT /settings read/write the `app_settings` store (execute tool toggle,
connection resolution policy). Mutations take effect immediately: the
resolved settings cache refreshes, cached agent graphs are dropped, and the
filesystem backend is rebuilt (execute <-> store backend swap), so the next
run picks up the new configuration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ....core.database import persistence
from ....core.dependencies import get_admin_user
from ....schema.settings_schema import SettingsIn, SettingsOut
from ....services import settings as settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


def _source(row: dict | None, field: str) -> str:
    """'db' when the setting row carries the field, else 'env'."""
    return "db" if row is not None and field in row else "env"


def _out() -> SettingsOut:
    execute_row = settings_service.get_setting("execute")
    connections_row = settings_service.get_setting("connections")
    return SettingsOut(
        execute={
            "enabled": settings_service.execute_enabled(),
            "max_timeout": settings_service.execute_max_timeout(),
            "inherit_env": settings_service.execute_inherit_env(),
            "source": _source(execute_row, "enabled"),
        },
        connections={
            "fallback_env": settings_service.connection_fallback_env(),
            "source": _source(connections_row, "fallback_env"),
        },
    )


async def _apply_mutation(request: Request) -> None:
    """Refresh the cache and rebuild agent graphs/backend for the new settings.

    Best-effort: the settings are already persisted; rebuilding the default
    graph may fail (e.g. env fallback disabled with no `llm` connection yet),
    in which case the next successful chat run rebuilds it.
    """
    import logging

    logger = logging.getLogger(__name__)
    await settings_service.refresh_app_settings()
    request.app.state.agents.update_persistence(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
    )
    request.app.state.backend = request.app.state.agents.backend
    try:
        request.app.state.agent = await request.app.state.agents.resolve("default", "anonymous")
    except Exception:
        logger.exception("agent rebuild after settings change failed; will retry on next run")


@router.get("", response_model=SettingsOut)
async def get_settings(_: dict = Depends(get_admin_user)):
    """Effective settings (DB values, else .env defaults)."""
    return _out()


@router.put("", response_model=SettingsOut)
async def put_settings(body: SettingsIn, request: Request, _: dict = Depends(get_admin_user)):
    """Upsert the provided setting keys; unset fields keep their current value."""
    if body.execute is not None:
        current = settings_service.get_setting("execute") or {}
        value = {
            "enabled": body.execute.enabled
            if body.execute.enabled is not None
            else current.get("enabled"),
            "max_timeout": body.execute.max_timeout
            if body.execute.max_timeout is not None
            else current.get("max_timeout"),
            "inherit_env": body.execute.inherit_env
            if body.execute.inherit_env is not None
            else current.get("inherit_env"),
        }
        await settings_service.update_app_setting("execute", value)
    if body.connections is not None:
        current = settings_service.get_setting("connections") or {}
        await settings_service.update_app_setting(
            "connections",
            {
                "fallback_env": body.connections.fallback_env
                if body.connections.fallback_env is not None
                else current.get("fallback_env"),
            },
        )
    await _apply_mutation(request)
    return _out()
