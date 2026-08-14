"""AI Backend API: Deep Agents harness behind a streaming (SSE) chat endpoint.

Thin app factory — routers live in `app/api/v1/endpoints/` (auth, chat, agent
resources, health), business logic in `app/services/`. The SSE event contract
below is normative — keep it in sync with `app/services/chat.py`.

Endpoints:
  POST /chat                      -> SSE stream of agent events (JSON body, or
                                    multipart/form-data with optional `files`
                                    uploads the agent reads/manipulates via its
                                    filesystem/execute tools)
  POST /api/chat                  -> AI SDK data-stream protocol (frontend useChat;
                                     supports HITL resume via `decision`/`decisions`)
  GET  /threads                   -> conversations for the current user (limit/offset)
  GET  /threads/{id}/messages     -> full message history of a thread
  GET  /threads/{id}/usage        -> context + usage report of a thread (session):
                                    message count/size, cumulative input/output/
                                    total tokens from usage_metadata, estimated
                                    API cost in USD (per-1M model rates), current
                                    context vs the model's window (utilization +
                                    remaining; utilization is a 0..1 fraction,
                                    display as a percentage), active-run flag
  POST /threads/{id}/title        -> LLM-generated title (prompt template) upserted
                                    on the thread metadata (create or update)
  POST /threads/{id}/followup     -> post-run follow-up (frontend): auto-title the
                                    thread unless an intentional title exists
                                    ({"force": true} regenerates) + up to 3
                                    suggested follow-up questions; returns
                                    {thread_id, title, generated, followups}
  PATCH /threads/{id}             -> rename a thread
  DELETE /threads/{id}            -> delete a thread
  POST /threads/{id}/resume       -> resume an interrupted (HITL) run
  POST /threads/{id}/cancel       -> abort the active run of a thread
  GET  /threads/{id}/stream       -> attach a live SSE stream to an active run
  GET  /notifications/stream      -> per-user run lifecycle events as SSE
                                    (run_started/completed/interrupted/
                                    cancelled/failed; recent events replayed on
                                    connect, ?since=<seq> skips seen ones)
  GET  /notifications             -> the user's recent run lifecycle events
  POST /threads/{id}/share        -> create a public share link (owner; idempotent)
  GET  /threads/{id}/share        -> current share link of a thread (owner)
  DELETE /threads/{id}/share      -> revoke the share link (owner)
  GET  /shared/{token}            -> public read-only view of a shared thread (no auth)
  POST /login                     -> JWT (access + refresh token)
  POST /refresh                   -> exchange a refresh token for a new access token
  GET  /users/me                  -> current user
  POST /users/me/password         -> change your own password
  GET  /users                     -> admin: list users
  POST /users                     -> admin: create a user
  PATCH /users/{username}         -> admin: change role/disabled state
  DELETE /users/{username}        -> admin: delete a user
  GET  /agent/skills|tools        -> agent resource CRUD (store-backed; skills
                                    include bundled files, e.g. scripts/;
                                    listing is readable by any user, mutations
                                    are admin-only)
  GET|POST /skills                -> user-scoped "my skills" CRUD (private per
                                    user, attachable to own agent configs)
  GET|POST /agents                -> list / create agent configs (customizable
                                    profiles: model + system prompt + skills +
                                    tools + temperature + thinking (reasoning
                                    effort: none..minimal..low..medium..high..
                                    xhigh..max); scope=global requires admin)
  GET|PUT|DELETE /agents/{name}   -> read / replace / delete an agent config
  POST /agents/{name}/test        -> dry-run: build the graph (validates model)
  POST /knowledge                -> create a knowledge base (per-user)
  GET  /knowledge                -> list the user's knowledge bases
  GET|PATCH|DELETE /knowledge/{id}-> detail / rename / delete a KB (vectors too)
  POST /knowledge/{id}/files     -> multipart upload (file + relative path per
                                    file; folder upload = many pairs). Each file
                                    is ingested: parse -> chunk -> embed -> Weaviate
  POST /knowledge/{id}/zip       -> upload a .zip of documents (safe extraction:
                                    traversal + zip-bomb guards, per-entry results)
  GET  /knowledge/{id}/files     -> documents with ingest status
  GET|DELETE /knowledge/{id}/files/{doc}-> document detail / delete (vectors too)
  GET  /knowledge/{id}/files/{doc}/content-> raw file bytes (inline preview)
  POST /knowledge/{id}/reindex   -> re-parse + re-embed all documents
  GET  /knowledge/{id}/search    -> hybrid (vector + keyword) search over a KB
                                    (optional ?alpha= 0..1 per request)
  GET  /knowledge/search         -> hybrid search across all the user's KBs
                                    (optional ?alpha= 0..1 per request)
                                    (the legacy /kb/* paths still answer too)
  POST /mcp/tools/call             -> prefab app tools proxy: invoke a tool on a
                                    configured MCP server (server_hint or fan-out;
                                    CallToolResult passthrough: 200 with isError,
                                    502 transport, 404 no match)
  GET|POST /connections            -> provider connections CRUD (admin; base URL +
                                    API token saved instead of .env; the agent LLM
                                    and KB embeddings resolve the default `llm` /
                                    `embeddings` connection)
  GET|PUT|DELETE /connections/{name}-> connection detail / replace / delete
  GET|PUT /settings                -> runtime app settings (admin; execute tool
                                    toggle + HITL interrupt_on + connection
                                    policy, DB overrides .env)
  GET  /health                    -> status

SSE events (event: <name>, data: <json>):
  message_delta   {"id", "delta"}                           token chunk
  reasoning_start {"id"}                                    thinking/reasoning turn started
  reasoning_delta {"id", "delta"}                           thinking/reasoning token chunk
  reasoning_end   {"id"}                                    thinking/reasoning turn finished
  message         {"id", "message"}                         finalized message (langchain serialized)
  tool_start      {"id", "name", "args"}                    tool call started
  tool_delta      {"id", "name", "delta"}                   tool output chunk
  tool_end        {"id", "name", "output", "is_error"}      tool call finished
  subagent        {"name", "status", "output"?, "error"?}   delegated task lifecycle
  subagent_delta  {"subagent", "delta"}                     subagent token chunk
  interrupt       {"thread_id", "interrupts"[{value, when}]} run paused for human input
  error           {"source", "message"}                     recoverable stream error
  done            {"thread_id", "messages"[], "interrupted"?,
                   "cancelled"?, "usage"?}                  final state (usage = summed
                                                             token counts when reported)

Run lifecycle: runs are decoupled from the HTTP stream — a client disconnect
("new chat") never aborts the run; it keeps processing in the background,
persists history + thread metadata, and emits a lifecycle event on the user's
notification stream. Only POST /threads/{id}/cancel aborts a run. Thread
metadata carries a `status` (running | completed | interrupted | cancelled |
failed) so GET /threads reflects in-flight runs.

Lifecycle events (notifications stream; data: {event_id, seq, thread_id,
title, agent, status, at}):
  run_started / run_completed / run_interrupted / run_cancelled / run_failed
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph.state import CompiledStateGraph

from .api.v1.routes import api_router
from .core.config import settings
from .core.database import persistence
from .services import settings as runtime_settings
from .services.agent import AgentRegistry, build_backend, build_extra_tools
from .services.connections import llm_model_name, refresh_resolved_connections
from .services.mcp import mcp_servers
from .services.reasoning_bridge import install as install_reasoning_bridge

logger = logging.getLogger(__name__)


def create_app(
    *,
    agent: CompiledStateGraph | None = None,
    agent_registry: AgentRegistry | None = None,
) -> FastAPI:
    # Lift `reasoning_content` deltas (DeepSeek-style providers) into reasoning
    # content blocks so thinking streams through ms.reasoning -> reasoning_delta.
    install_reasoning_bridge()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await persistence.start()
        # Resolve saved provider connections (default llm/embeddings) so the
        # agent model + KB embeddings use them instead of .env credentials.
        await refresh_resolved_connections()
        # Load runtime app settings (execute tool toggle, connection policy)
        # so DB values override .env from the first request on.
        await runtime_settings.refresh_app_settings()
        # Shared durable filesystem backend: agent and /agent/* CRUD API write
        # to the same store (Postgres in production), so skills added via the
        # API are visible to the agent on the next run.
        app.state.backend = build_backend(store=persistence.store)
        try:
            await mcp_servers.connect(store=persistence.store)
        except Exception:
            logger.exception("MCP connect failed; continuing without MCP tools")
        search_tool = build_extra_tools()  # list: web_search + kb search tool (when configured)
        # Agent registry: lazy graph factory keyed by agent config (model +
        # system prompt + skills + tools). `agent` (tests) becomes the static
        # default; every resolve() then returns that graph.
        app.state.agents = agent_registry or AgentRegistry(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            backend=app.state.backend,
            mcp_tools=mcp_servers.tools,
            extra_tools=search_tool or None,
            tools_by_server=mcp_servers.tools_by_server,
            static_default=agent,
        )
        # Registries injected from outside (tests) may hold pre-start
        # checkpointer/store references — rebind to the live instances.
        app.state.agents.update_persistence(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
        )
        app.state.backend = app.state.agents.backend
        app.state.agent = await app.state.agents.resolve("default", "anonymous")
        logger.info(
            "Agent ready: model=%s persistence=%s mcp=%s searxng=%s execute=%s",
            settings.model or llm_model_name(),
            persistence.backend_name,
            mcp_servers.names or "none",
            "enabled" if search_tool else "not configured",
            "enabled" if runtime_settings.execute_enabled() else "disabled",
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
