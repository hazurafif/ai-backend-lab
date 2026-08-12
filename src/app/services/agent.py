"""Deep Agent factory: create_deep_agent wired with persistence + MCP tools.

The agent gets the Deep Agents built-in harness:
  - filesystem tools (ls, read_file, write_file, edit_file, grep, glob, execute, ...)
  - `task` tool to delegate to subagents (isolated context windows)
  - context management (summarization, offloading)
  - skills / memory via AGENTS.md-style files

Plus:
  - MCP tools from gofastmcp servers (passed via `tools=`)
  - Postgres checkpointer (conversations) + Postgres store (cross-thread memory)
  - durable filesystem backend: by default a StoreBackend over the LangGraph
    store (Postgres in production, in-memory in dev), so every user's workspace
    persists across threads. `/memories/` (per user) and `/skills/` (global,
    shared by all users) are routed to the store as well.
  - skills loaded from the store-backed `/skills/` source, managed via the
    /agent/skills CRUD API (no agent rebuild needed — SkillsMiddleware reads
    the backend on every run).
"""

from __future__ import annotations

import logging
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..core.config import settings
from ..core.constants import GLOBAL_SKILLS_NS, SKILLS_SOURCE
from ..services.kb.tool import build_kb_search_tool
from ..services.searxng import build_search_tool

logger = logging.getLogger(__name__)


def build_extra_tools() -> list[BaseTool]:
    """Optional agent tools: web search (SearXNG) + knowledge base search.

    Each tool is only included when its backend is configured (SEARXNG_URL /
    WEAVIATE_URL), so an unconfigured service never registers a dead tool.
    """
    tools: list[BaseTool] = []
    for candidate in (build_search_tool, build_kb_search_tool):
        tool_ = candidate()
        if tool_ is not None:
            tools.append(tool_)
    return tools


def _user_namespace_factory(rt: Any) -> tuple[str, ...]:
    """Scope store-backed workspace/memory files per user.

    `rt` is the runtime context object; it carries per-run `context` data
    (e.g. user_id) passed at invocation time.
    """
    ctx = getattr(rt, "context", None) or {}
    if isinstance(ctx, dict):
        user = ctx.get("user_id", "anonymous")
    else:
        user = getattr(ctx, "user_id", "anonymous")
    return (str(user),)


def _global_namespace_factory(_rt: Any) -> tuple[str, ...]:
    """Global namespace shared by all users (agent-level resources)."""
    return GLOBAL_SKILLS_NS


def build_backend(*, store: BaseStore) -> CompositeBackend:
    """Build the shared filesystem backend for the agent and the resources API.

    The default backend is a durable StoreBackend over the LangGraph store
    (Postgres in production), so every user's workspace persists across
    threads. With `EXECUTE_ENABLED=true` the default is LocalShellBackend
    (host shell) instead. `/memories/` (per user) and `/skills/` (global) are
    always routed to the durable store.
    """
    user_store = StoreBackend(store=store, namespace=_user_namespace_factory)
    default = (
        LocalShellBackend(inherit_env=settings.execute_inherit_env)
        if settings.execute_enabled
        else user_store
    )
    return CompositeBackend(
        default=default,
        routes={
            "/memories/": user_store,
            "/skills/": StoreBackend(store=store, namespace=_global_namespace_factory),
        },
    )


def build_agent(
    *,
    checkpointer: Checkpointer,
    store: BaseStore,
    mcp_tools: list[BaseTool] | None = None,
    extra_tools: list[BaseTool] | None = None,
    backend: CompositeBackend | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    interrupt_on: dict[str, Any] | None = None,
) -> CompiledStateGraph:
    """Build the core Deep Agent.

    Args:
        checkpointer: persists conversation state per thread_id (Postgres or in-memory).
        store: long-term memory store (Postgres or in-memory).
        mcp_tools: tools loaded from MCP servers (gofastmcp).
        extra_tools: additional tools, e.g. the SearXNG web_search tool.
        backend: shared filesystem backend; created from the store when omitted.
            Pass the lifespan-built instance so the /agent/skills CRUD API and
            the agent see the same files.
        model: provider:model string, e.g. "openai:gpt-4o-mini".
        system_prompt: agent instructions.
        interrupt_on: {"tool_name": True} to pause for human approval.
    """
    backend = backend or build_backend(store=store)

    # Replace the default FilesystemMiddleware only when execution is enabled,
    # so EXECUTE_MAX_TIMEOUT applies to the execute tool's per-command cap.
    middleware = (
        [FilesystemMiddleware(backend=backend, max_execute_timeout=settings.execute_max_timeout)]
        if settings.execute_enabled
        else None
    )

    tools = list(mcp_tools or []) + list(extra_tools or [])

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
        backend=backend,
        middleware=middleware or (),
        skills=[SKILLS_SOURCE],
        interrupt_on=interrupt_on or None,
    )
    logger.info(
        "Deep agent built (model=%s, %d tools, execute=%s)",
        model,
        len(tools),
        "enabled" if settings.execute_enabled else "disabled",
    )
    return agent
