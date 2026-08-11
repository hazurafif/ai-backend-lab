"""Health route: /health."""

from __future__ import annotations

from fastapi import APIRouter

from ..core.config import settings
from ..db import persistence
from ..services import resources
from ..services.mcp import mcp_servers

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "persistence": persistence.backend_name,
        "mcp_servers": mcp_servers.names,
        "model": settings.model,
        "interrupt_on": settings.interrupt_on,
        "searxng": {
            "installed": settings.searxng_url is not None,
            "enabled": settings.searxng_enabled,
        },
        "execute": {
            "enabled": settings.execute_enabled,
            "max_timeout": settings.execute_max_timeout,
        },
        "agent_resources": {
            "skills": len(await resources.list_skills(persistence.store)),
            "tool_servers": len(await resources.list_tool_servers(persistence.store)),
        },
    }
