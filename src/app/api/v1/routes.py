"""API v1 router aggregation (per-domain endpoint modules)."""

from fastapi import APIRouter

from app.api.v1.endpoints.agent import router as agent_router
from app.api.v1.endpoints.agents import router as agents_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.health import router as health_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(agent_router)
api_router.include_router(agents_router)
api_router.include_router(health_router)
