"""AI Backend API: Deep Agents harness behind a streaming (SSE) chat endpoint.

Thin app factory — routers live in `app/api/` (auth, chat, agent resources,
health), business logic in `app/services/`. The SSE event contract below is
normative — keep it in sync with `app/services/chat.py`.

Endpoints:
  POST /chat                      -> SSE stream of agent events
  POST /api/chat                  -> AI SDK data-stream protocol (frontend useChat)
  GET  /threads                   -> conversations for the current user
  GET  /threads/{id}/messages     -> full message history of a thread
  POST /threads/{id}/resume       -> resume an interrupted (HITL) run
  POST /login                     -> JWT
  GET  /users/me                  -> current user
  GET  /agent/skills|tools        -> agent resource CRUD (store-backed)
  GET  /health                    -> status

SSE events (event: <name>, data: <json>):
  message_delta   {"id", "delta"}                           token chunk
  message         {"id", "message"}                         finalized message (langchain serialized)
  tool_start      {"id", "name", "args"}                    tool call started
  tool_delta      {"id", "name", "delta"}                   tool output chunk
  tool_end        {"id", "name", "output", "is_error"}      tool call finished
  subagent        {"name", "status", "output"?, "error"?}   delegated task lifecycle
  subagent_delta  {"subagent", "delta"}                     subagent token chunk
  interrupt       {"thread_id", "interrupts"[{value, when}]} run paused for human input
  error           {"source", "message"}                     recoverable stream error
  done            {"thread_id", "messages"[], "interrupted"?} final state
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph.state import CompiledStateGraph

from .api import api_router
from .core.config import settings
from .db import persistence
from .services.agent import build_agent, build_backend
from .services.mcp import mcp_servers
from .services.searxng import build_search_tool

logger = logging.getLogger(__name__)


def create_app(*, agent: CompiledStateGraph | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await persistence.start()
        # Shared durable filesystem backend: agent and /agent/* CRUD API write
        # to the same store (Postgres in production), so skills added via the
        # API are visible to the agent on the next run.
        app.state.backend = build_backend(store=persistence.store)
        try:
            await mcp_servers.connect(store=persistence.store)
        except Exception:
            logger.exception("MCP connect failed; continuing without MCP tools")
        search_tool = build_search_tool()
        app.state.agent = agent or build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            mcp_tools=mcp_servers.tools,
            extra_tools=[search_tool] if search_tool else None,
            backend=app.state.backend,
            model=settings.model,
            system_prompt=settings.system_prompt,
            interrupt_on=settings.interrupt_on,
        )
        logger.info(
            "Agent ready: model=%s persistence=%s mcp=%s searxng=%s execute=%s",
            settings.model,
            persistence.backend_name,
            mcp_servers.names or "none",
            "enabled" if search_tool else "not configured",
            "enabled" if settings.execute_enabled else "disabled",
        )
        yield
        await mcp_servers.close()
        await persistence.stop()

    app = FastAPI(title="AI Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
