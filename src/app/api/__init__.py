"""API layer: routers + shared dependencies (per-domain modules)."""

from fastapi import APIRouter

from . import routes_agent, routes_auth, routes_chat, routes_health

api_router = APIRouter()
api_router.include_router(routes_auth.router)
api_router.include_router(routes_chat.router)
api_router.include_router(routes_agent.router)
api_router.include_router(routes_health.router)
