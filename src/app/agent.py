"""Deep Agent factory: create_deep_agent wired with persistence + MCP tools.

The agent gets the Deep Agents built-in harness:
  - filesystem tools (ls, read_file, write_file, edit_file, grep, glob, ...)
  - `task` tool to delegate to subagents (isolated context windows)
  - context management (summarization, offloading)
  - skills / memory via AGENTS.md-style files

Plus:
  - MCP tools from gofastmcp servers (passed via `tools=`)
  - Postgres checkpointer (conversations) + Postgres store (cross-thread memory)
  - store-backed `/memories/` filesystem backend scoped per user
"""

from __future__ import annotations

import logging
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.store import StoreBackend
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)


def _user_namespace_factory(rt: Any) -> tuple[str, ...]:
    """Scope store-backed memory files per user.

    `rt` is the runtime context object; it carries per-run `context` data
    (e.g. user_id) passed at invocation time.
    """
    ctx = getattr(rt, "context", None) or {}
    if isinstance(ctx, dict):
        user = ctx.get("user_id", "anonymous")
    else:
        user = getattr(ctx, "user_id", "anonymous")
    return (str(user),)


def build_agent(
    *,
    checkpointer: Checkpointer,
    store: BaseStore,
    mcp_tools: list[BaseTool] | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    interrupt_on: dict[str, Any] | None = None,
) -> CompiledStateGraph:
    """Build the core Deep Agent.

    Args:
        checkpointer: persists conversation state per thread_id (Postgres or in-memory).
        store: long-term memory store (Postgres or in-memory).
        mcp_tools: tools loaded from MCP servers (gofastmcp).
        model: provider:model string, e.g. "openai:gpt-4o-mini".
        system_prompt: agent instructions.
        interrupt_on: {"tool_name": True} to pause for human approval.
    """
    # Filesystem backend: thread-scoped scratch space by default, but
    # `/memories/` is routed to the persistent store so memory survives
    # across conversations, scoped per user.
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(store=store, namespace=_user_namespace_factory),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=list(mcp_tools or []),
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
        backend=backend,
        interrupt_on=interrupt_on or None,
    )
    logger.info("Deep agent built (model=%s, %d MCP tools)", model, len(mcp_tools or []))
    return agent
