"""Setup routes: per-user startup state + onboarding.

GET /users/me/setup    -> the user's setup state: admin-managed llm
                         connection (read-only, masked) + effective model +
                         the user's own MCP servers + preferences. A frontend
                         shows a startup screen until `completed`.
POST /users/me/onboarding -> one-shot setup for per-user data: preferences
                         and MCP tool servers (idempotent upserts).

Connections (the completions API) are **admin-only**: create/manage them via
POST /connections (admin). Users never submit or see credentials.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from ....core.config import settings
from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....schema.preferences_schema import PreferencesIn, PreferencesOut
from ....schema.setup_schema import OnboardingIn, SetupOut
from ....services import agent_configs, connections, resources
from ....services.mcp import mcp_servers

logger = logging.getLogger(__name__)

router = APIRouter(tags=["setup"])


async def _effective_preferences(username: str) -> PreferencesOut:
    """Stored preferences merged over the server defaults (stored wins)."""
    stored = await persistence.preferences.get_all(username)
    defaults = {
        "enable_search": settings.searxng_enabled,
        "hide_reasoning": False,
        "hide_tool_calls": False,
    }
    return PreferencesOut(**{key: stored.get(key, default) for key, default in defaults.items()})


async def _setup_out(username: str) -> SetupOut:
    """The user's setup state (connection info is the admin-managed default)."""
    conn = connections.resolved_llm()
    model = settings.model or connections.llm_model_name()
    agent = await agent_configs.get_config(persistence.store, "default", username)
    servers = await resources.list_tool_servers(persistence.store, username)
    return SetupOut(
        completed=conn is not None and bool(conn.get("api_token")) and bool(model),
        llm_connection=connections.to_out(conn) if conn is not None else None,
        model=model,
        agent=agent,
        mcp_servers=servers,
        preferences=await _effective_preferences(username),
    )


@router.get("/users/me/setup", response_model=SetupOut)
async def read_setup(current_user: dict = Depends(get_current_user)):
    """The user's setup state (frontend renders the startup screen from it)."""
    return await _setup_out(current_user["username"])


@router.post("/users/me/onboarding", response_model=SetupOut, status_code=201)
async def onboarding(
    body: OnboardingIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """One-shot setup of the user's per-user data: preferences + MCP servers.

    Idempotent: re-running updates the stored preferences and upserts MCP
    servers. The completions API itself is admin-managed (POST /connections).
    """
    username = current_user["username"]
    if body.preferences is not None:
        for key in PreferencesIn.model_fields:
            if key in body.preferences.model_fields_set:
                await persistence.preferences.set(username, key, getattr(body.preferences, key))
    for server in body.mcp_servers:
        try:
            await resources.create_tool_server(persistence.store, username, server)
        except KeyError:
            await resources.update_tool_server(persistence.store, username, server.name, server)
    try:
        await mcp_servers.connect(username, persistence.store)
    except Exception:
        logger.exception("MCP connect after onboarding failed for %s", username)
    request.app.state.agents.invalidate()
    return await _setup_out(username)
