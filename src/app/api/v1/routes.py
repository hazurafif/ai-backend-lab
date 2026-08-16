"""API v1 router aggregation (per-domain endpoint modules)."""

from fastapi import APIRouter

from app.api.v1.endpoints.agent import router as agent_router
from app.api.v1.endpoints.agents import router as agents_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.connections import router as connections_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.kb import router as kb_router
from app.api.v1.endpoints.mcp import router as mcp_router
from app.api.v1.endpoints.preferences import router as preferences_router
from app.api.v1.endpoints.settings import router as settings_router
from app.api.v1.endpoints.setup import router as setup_router
from app.api.v1.endpoints.skills import router as skills_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(connections_router)
api_router.include_router(agent_router)
api_router.include_router(agents_router)
api_router.include_router(skills_router)
api_router.include_router(kb_router, prefix="/knowledge")
# Backward-compatible alias: the KB routes also answer under /kb.
api_router.include_router(kb_router, prefix="/kb")
api_router.include_router(mcp_router)
api_router.include_router(preferences_router)
api_router.include_router(setup_router)
api_router.include_router(settings_router)
api_router.include_router(health_router)
