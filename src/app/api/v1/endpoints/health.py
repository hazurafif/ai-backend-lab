"""Health route: /health."""

from __future__ import annotations

from fastapi import APIRouter

from ....core.config import settings
from ....core.database import persistence
from ....services import resources
from ....services import settings as runtime_settings
from ....services.connections import llm_model_name, resolved_llm
from ....services.mcp import mcp_servers

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    # MCP servers are per user now: report the default admin's instance as
    # the infra summary. Connections (llm/embeddings) are admin-managed and
    # global — reported from the resolved cache.
    admin = settings.default_admin_username
    instance = await mcp_servers.get(admin, persistence.store)
    return {
        "status": "ok",
        "persistence": persistence.backend_name,
        "mcp_servers": instance.names,
        "model": settings.model or llm_model_name(),
        "llm_connection": (resolved_llm() or {}).get("name"),
        "llm_connection_model": llm_model_name(),
        "interrupt_on": runtime_settings.interrupt_on(),
        "searxng": {
            "installed": settings.searxng_url is not None,
            "enabled": settings.searxng_enabled,
        },
        "execute": {
            "enabled": runtime_settings.execute_enabled(),
            "max_timeout": runtime_settings.execute_max_timeout(),
        },
        "agent_resources": {
            "skills": len(await resources.list_skills(persistence.store)),
            "tool_servers": len(await resources.list_tool_servers(persistence.store, admin)),
        },
    }
