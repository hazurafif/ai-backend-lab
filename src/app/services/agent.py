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
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..core.config import settings
from ..core.constants import GLOBAL_SKILLS_NS, SKILLS_SOURCE
from ..services import settings as runtime_settings
from ..services.connections import llm_model_kwargs
from ..services.kb.tool import build_kb_search_tool
from ..services.searxng import build_search_tool
from .agent_configs import AgentSpec, load_spec

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


def build_backend(
    *,
    store: BaseStore,
    agent_skill_routes: dict[str, tuple[str, ...]] | None = None,
) -> CompositeBackend:
    """Build the shared filesystem backend for the agent and the resources API.

    The default backend is a durable StoreBackend over the LangGraph store
    (Postgres in production), so every user's workspace persists across
    threads. With `EXECUTE_ENABLED=true` the default is LocalShellBackend
    (host shell) instead. `/memories/` (per user), `/skills/` (global),
    `/tmp/` (per user) and `/uploads/` (per user) are always routed to the
    durable store — so even in execute mode, filesystem-middleware writes
    under those paths persist to the database, not the host filesystem.
    (Shell commands via the `execute` tool bypass backend routing and always
    see the real host filesystem.)

    `agent_skill_routes` maps skills source paths (e.g. `/skills/alice/research/`)
    to store namespaces for named agents; longer prefixes win, so these
    shadow `/skills/` for their own paths only.
    """
    user_store = StoreBackend(store=store, namespace=_user_namespace_factory)
    default = (
        LocalShellBackend(inherit_env=runtime_settings.execute_inherit_env())
        if runtime_settings.execute_enabled()
        else user_store
    )
    routes: dict[str, StoreBackend] = {
        "/memories/": user_store,
        "/skills/": StoreBackend(store=store, namespace=_global_namespace_factory),
        # Scratch + uploads stay durable even when the default is the host
        # shell: agent file-tool writes under these paths land in the store.
        "/tmp/": user_store,
        "/uploads/": user_store,
    }
    for source, ns in (agent_skill_routes or {}).items():
        routes[source] = StoreBackend(store=store, namespace=lambda rt, ns=ns: ns)
    return CompositeBackend(default=default, routes=routes)


def build_agent(
    *,
    checkpointer: Checkpointer,
    store: BaseStore,
    mcp_tools: list[BaseTool] | None = None,
    extra_tools: list[BaseTool] | None = None,
    backend: CompositeBackend | None = None,
    model: str | BaseChatModel | None = None,
    system_prompt: str | None = None,
    interrupt_on: dict[str, Any] | None = None,
    skills: list[str] | None = None,
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
        model: provider:model string, e.g. "openai:gpt-4o-mini", or an already
            constructed chat model (tests).
        system_prompt: agent instructions.
        interrupt_on: {"tool_name": True} to pause for human approval.
        skills: SkillsMiddleware source paths (e.g. the per-agent
            "/skills/<owner>/<name>/" route). None = the global "/skills/"
            source; [] = no skills middleware.
    """
    backend = backend or build_backend(store=store)

    # Replace the default FilesystemMiddleware only when execution is enabled,
    # so the runtime execute max-timeout applies to the execute tool's
    # per-command cap.
    middleware = (
        [
            FilesystemMiddleware(
                backend=backend, max_execute_timeout=runtime_settings.execute_max_timeout()
            )
        ]
        if runtime_settings.execute_enabled()
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
        skills=skills if skills is not None else [SKILLS_SOURCE],
        interrupt_on=interrupt_on or None,
    )
    logger.info(
        "Deep agent built (model=%s, %d tools, execute=%s)",
        model,
        len(tools),
        "enabled" if runtime_settings.execute_enabled() else "disabled",
    )
    return agent


class AgentRegistry:
    """Lazy agent factory: builds and caches compiled graphs per agent config.

    The deep-agents graph bakes the model + system prompt + skills sources in
    at build time, so per-request customization means building one graph per
    distinct config. Graphs share the same checkpointer/store/backend, so
    conversation state survives rebuilds. The cache is keyed by a fingerprint
    of the resolved spec; `invalidate()` (after skills/tools/config CRUD)
    drops it so the next request rebuilds.
    """

    def __init__(
        self,
        *,
        checkpointer: Checkpointer,
        store: BaseStore,
        backend: CompositeBackend,
        mcp_tools: list[BaseTool] | None = None,
        extra_tools: list[BaseTool] | None = None,
        tools_by_server: dict[str, list[str]] | None = None,
        model_factory: Callable[[str, float | None], str | BaseChatModel] | None = None,
        static_default: CompiledStateGraph | None = None,
        max_cache: int = 16,
    ) -> None:
        """Args:
        checkpointer/store/backend: shared persistence + filesystem backend
            (one instance for every graph).
        mcp_tools: tools loaded from MCP servers.
        extra_tools: additional tools, e.g. the SearXNG web_search tool.
        tools_by_server: tool name -> MCP server name attribution, for
            per-agent tool selection (`web_search` is a built-in pseudo-tool).
        model_factory: test hook returning a chat model for (model, temperature).
        static_default: when set (tests), every resolve() returns this graph.
        """
        self._checkpointer = checkpointer
        self._store = store
        self._backend = backend
        self._mcp_tools = list(mcp_tools or [])
        self._extra_tools = list(extra_tools or [])
        self._tools_by_server = dict(tools_by_server or {})
        self._model_factory = model_factory
        self._static_default = static_default
        self._cache: OrderedDict[str, CompiledStateGraph] = OrderedDict()
        self._max_cache = max_cache
        self._agent_skill_routes: dict[str, tuple[str, ...]] = {}

    @property
    def backend(self) -> CompositeBackend:
        """Current filesystem backend (rebuilt when a new agent route appears)."""
        return self._backend

    async def resolve(
        self, name: str = "default", username: str = "anonymous"
    ) -> CompiledStateGraph:
        """The compiled graph for an agent config (user -> global -> builtin default).

        Raises KeyError when the config does not exist.
        """
        if self._static_default is not None:
            return self._static_default
        spec = await load_spec(self._store, name, username)
        if spec is None:
            raise KeyError(name)
        return await self._build_for(spec)

    async def _build_for(self, spec: AgentSpec) -> CompiledStateGraph:
        key = spec.fingerprint()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        self._ensure_skill_route(spec)
        tools = self._select_tools(spec)
        model = self._resolve_model(spec)
        graph = build_agent(
            checkpointer=self._checkpointer,
            store=self._store,
            mcp_tools=tools,
            backend=self._backend,
            model=model,
            system_prompt=spec.system_prompt or settings.system_prompt,
            interrupt_on=spec.interrupt_on or settings.interrupt_on,
            skills=[spec.skills_source]
            if spec.skills_source
            else ([] if spec.skills is not None else None),
        )
        self._cache[key] = graph
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return graph

    def _ensure_skill_route(self, spec: AgentSpec) -> None:
        """Add the agent's skills backend route (rebuilding the backend if new)."""
        source = spec.skills_source
        ns = spec.skills_ns
        if source is None or ns is None or source in self._backend.routes:
            return
        self._agent_skill_routes[source] = ns
        self._backend = build_backend(
            store=self._store, agent_skill_routes=self._agent_skill_routes
        )
        self._cache.clear()
        logger.info("agent skills route added: %s -> %s", source, ns)

    def _select_tools(self, spec: AgentSpec) -> list[BaseTool]:
        """Tools for the agent: inherited (all) or the named selection."""
        if spec.tools is None:
            return list(self._mcp_tools) + list(self._extra_tools)
        selected: list[BaseTool] = []
        by_name = {t.name: t for t in self._mcp_tools}
        for name in spec.tools:
            if name == "web_search":
                selected.extend(t for t in self._extra_tools if t.name == "web_search")
                continue
            for tool_name in self._tools_by_server.get(name, []):
                if tool_name in by_name and by_name[tool_name] not in selected:
                    selected.append(by_name[tool_name])
        return selected

    def _resolve_model(self, spec: AgentSpec) -> str | BaseChatModel:
        if self._model_factory is not None:
            return self._model_factory(spec.model, spec.temperature)
        # A saved `llm` connection (base URL + API token, see /connections)
        # overrides .env credentials for the provider. When no default llm
        # connection exists, env fallback is opt-in (PUT /settings
        # connections.fallback_env=true); otherwise this fails loudly so the
        # agent never silently runs on .env credentials.
        kwargs = llm_model_kwargs()
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.thinking is not None:
            kwargs["reasoning_effort"] = spec.thinking
        if not kwargs:
            if not runtime_settings.connection_fallback_env():
                raise ValueError(
                    "No default 'llm' connection configured — create one via "
                    "POST /connections (kind=llm, is_default=true), or allow env "
                    'credentials via PUT /settings {"connections": '
                    '{"fallback_env": true}}'
                )
            return spec.model
        from langchain.chat_models import init_chat_model

        model = init_chat_model(spec.model, **kwargs)
        # Models without a langchain model profile (e.g. deepseek-v4-flash via
        # the Console Go gateway) come back with profile=None, and deepagents'
        # FilesystemMiddleware treats missing profile fields as "supported" —
        # so read_file media blocks are forwarded to the provider, which 400s
        # ("unknown variant `image_url`, expected `text`"). Declare the
        # text-only capabilities the gateway actually supports so unsupported
        # media is scrubbed to a placeholder note. Known vision models keep
        # their declared capabilities (only missing fields are filled).
        profile = dict(model.profile or {})
        for field in ("image_inputs", "audio_inputs", "video_inputs", "pdf_inputs"):
            profile.setdefault(field, False)
        model.profile = profile
        return model

    async def model_for(self, name: str, username: str) -> str | BaseChatModel:
        """The chat model of an agent config (spec resolution + model build).

        Used by auxiliary LLM calls (e.g. thread title generation) that want
        the thread's own model without running the full agent graph.
        """
        spec = await load_spec(self._store, name, username)
        if spec is None:
            raise KeyError(name)
        return self._resolve_model(spec)

    def update_mcp_tools(
        self,
        mcp_tools: list[BaseTool] | None,
        tools_by_server: dict[str, list[str]] | None = None,
    ) -> None:
        """Replace the MCP tool set (after reconnect) and drop all cached graphs."""
        self._mcp_tools = list(mcp_tools or [])
        if tools_by_server is not None:
            self._tools_by_server = dict(tools_by_server)
        self.invalidate()

    def update_extra_tools(self, extra_tools: list[BaseTool] | None) -> None:
        """Replace the non-MCP tools (web_search, kb search, ...) and drop cached graphs."""
        self._extra_tools = list(extra_tools or [])
        self.invalidate()

    def update_persistence(self, *, checkpointer: Checkpointer, store: BaseStore) -> None:
        """Rebind persistence after `persistence.start()` (lifespan), e.g. for
        registries constructed before startup with an in-memory fallback.
        The filesystem backend is rebuilt on the new store (agent skill routes
        preserved) and cached graphs are dropped.
        """
        self._checkpointer = checkpointer
        self._store = store
        self._backend = build_backend(store=store, agent_skill_routes=self._agent_skill_routes)
        self._cache.clear()

    def invalidate(self) -> None:
        """Drop cached graphs; the next request rebuilds from current resources."""
        self._cache.clear()
