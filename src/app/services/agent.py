"""Deep Agent factory: create_deep_agent wired with persistence + MCP tools.

The agent gets the Deep Agents built-in harness:
  - filesystem tools (ls, read_file, write_file, edit_file, grep, glob, execute, ...)
  - `task` tool to delegate to subagents (isolated context windows)
  - context management (summarization, offloading)
  - skills / memory via AGENTS.md-style files

Plus:
  - MCP tools from gofastmcp servers (passed via `tools=`)
  - Postgres checkpointer (conversations) + Postgres store (agent configs,
    skills, thread metadata)
  - one real per-user filesystem: every file-tool path and shell command
    resolves to ``WORKSPACE_ROOT/<user_id>/`` (plain host files, bind-mounted
    in compose, git-versioned by services/workspace). No virtual mounts, no
    store mirroring — the disk is the source of truth.
  - skills materialized into the user's workspace before each run (see
    services/workspace.materialize_skills), loaded via the store-backed
    /skills CRUD API (no agent rebuild needed — SkillsMiddleware reads the
    backend on every run).

The agent is told its workspace layout (skills/, uploads/, tmp/, memories/)
by the system prompt — see DEFAULT_SYSTEM_PROMPT in core/constants.py, which
is rendered per user ({{username}}) so each agent only knows its own dir.
"""

from __future__ import annotations

import logging
import subprocess
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse, GlobResult
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import get_runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.constants import (
    GLOBAL_SKILLS_NS,
    MEMORY_SOURCE,
    SKILLS_SOURCE,
    user_skills_ns,
)
from ..core.database import persistence
from ..schema.agent_schema import SkillFileIn, SkillIn
from ..services import resources
from ..services import settings as runtime_settings
from ..services.connections import llm_model_kwargs, llm_model_name, model_kwargs_from
from ..services.kb.tool import build_kb_search_tool
from ..services.searxng import build_fetch_page_tool, build_search_tool
from .agent_configs import AgentSpec, load_spec

logger = logging.getLogger(__name__)


class PublishSkillInput(BaseModel):
    """Arguments for the `publish_skill` tool."""

    skill_path: str = Field(
        ...,
        description=(
            "Workspace path (virtual, e.g. '/scripts/my-skill' or '/tmp/draft-skill') "
            "of the folder containing SKILL.md to publish. SKILL.md must start with "
            "a YAML frontmatter block carrying 'name' and 'description'; every other "
            "file in the folder is bundled with the skill."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Replace an existing skill of the same name. Without it, publishing "
            "over an existing name is an error."
        ),
    )


def _parse_skill_frontmatter(text: str) -> tuple[str, str, str]:
    """(name, description, body) from a SKILL.md with YAML frontmatter.

    Raises ValueError when the frontmatter is malformed or misses the
    required `name` / `description` keys (single-line values only).
    """
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with a YAML frontmatter block ('---')")
    end = text.find("\n---", 3)
    if end == -1:
        raise ValueError("SKILL.md frontmatter is not closed (missing trailing '---')")
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-", ":")):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        meta[key.strip()] = value.strip().strip("\"'")
    name = meta.get("name", "")
    description = meta.get("description", "")
    if not name:
        raise ValueError("SKILL.md frontmatter is missing the required 'name' field")
    if not description:
        raise ValueError("SKILL.md frontmatter is missing the required 'description' field")
    return name, description, text[end + 4 :].strip("\n")


_SKIP_BUNDLE_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def _resolve_user_folder(path: str) -> Path:
    """Resolve a virtual workspace path under the current user's dir (file-tool rules)."""
    vpath = path if path.startswith("/") else "/" + path
    if ".." in vpath or vpath.startswith("~"):
        raise ValueError("Path traversal not allowed")
    base = (Path(settings.workspace_root) / _runtime_user_id()).resolve()
    full = (base / vpath.lstrip("/")).resolve()
    try:
        full.relative_to(base)
    except ValueError:
        raise ValueError(f"Path:{path} outside the workspace root") from None
    return full


def _bundle_files(folder: Path) -> list[SkillFileIn]:
    """All files under the skill folder (relative paths), minus SKILL.md and junk dirs."""
    files: list[SkillFileIn] = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink() or path.is_dir():
            continue
        rel = path.relative_to(folder).as_posix()
        if rel == "SKILL.md":
            continue
        if any(part in _SKIP_BUNDLE_DIRS for part in path.parts[len(folder.parts) :]):
            continue
        files.append(
            SkillFileIn(
                path=rel,
                content=path.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return files


async def _publish_skill(skill_path: str, overwrite: bool = False) -> str:
    """Tool body: publish the SKILL.md folder at `skill_path` into the user's store.

    Goes through the same `resources.create_skill` code path as the REST API,
    so a published skill shows up in the frontend skills list and is
    materialized into the user's workspace skills/ on every future run.
    """
    try:
        folder = _resolve_user_folder(skill_path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not folder.is_dir():
        return f"Error: no folder at '{skill_path}' — create it first (SKILL.md + bundled files)."
    skill_md = folder / "SKILL.md"
    if not skill_md.is_file():
        return (
            f"Error: no SKILL.md in '{skill_path}' — the folder must contain a "
            "SKILL.md with YAML frontmatter (name + description)."
        )
    try:
        name, description, body = _parse_skill_frontmatter(
            skill_md.read_text(encoding="utf-8", errors="replace")
        )
    except ValueError as exc:
        return f"Error: {exc}"
    try:
        files = _bundle_files(folder)
    except ValueError as exc:
        return f"Error: {exc}"  # e.g. a bundled file path failing the skill schema
    store = persistence.store
    if store is None:
        return "Error: persistence store is not available"
    ns = user_skills_ns(_runtime_user_id())
    if await resources.get_skill(store, name, ns) is not None and not overwrite:
        return (
            f"Error: skill '{name}' already exists — pass overwrite=true to replace "
            "it, or pick another name."
        )
    try:
        out = await resources.create_skill(
            store, SkillIn(name=name, description=description, content=body, files=files), ns
        )
    except ValueError as exc:
        return f"Error: {exc}"
    note = f" ({len(out.files)} bundled file(s))" if out.files else ""
    return (
        f"Published skill '{out.name}'{note}. It is now listed in the user's "
        "frontend skills library and is loaded into skills/ on every future run."
    )


def build_skill_publish_tool() -> BaseTool:
    """The `publish_skill` tool: persist a drafted skill into the user's store.

    The agent drafts a skill folder in its workspace (SKILL.md + bundled
    files) and calls this tool to publish it through the same `resources`
    code path as the /skills REST API — it appears in the frontend skills
    list immediately and survives the per-run workspace refresh.
    """
    return StructuredTool.from_function(
        coroutine=_publish_skill,
        name="publish_skill",
        description=(
            "Publish a skill to the user's skill library. The skill must be a "
            "folder in your workspace containing SKILL.md with YAML frontmatter "
            "(name + description); every other file in the folder is bundled "
            "with it. The skill is stored server-side: it appears in the "
            "frontend skills list immediately and is loaded into your skills/ "
            "folder at the start of every future run. Use this when the user "
            "asks you to create a skill."
        ),
        args_schema=PublishSkillInput,
    )


# Agent tools that are built in (not MCP servers): selectable by name in
# per-agent tool lists without a matching MCP server.
BUILTIN_PSEUDO_TOOLS = ("web_search", "fetch_page")


def build_extra_tools() -> list[BaseTool]:
    """Agent tools: skill publishing + optional web search / knowledge base search.

    `publish_skill` is always registered (core capability — the only way the
    agent can persist a new skill into the store). The search tools are only
    included when their backend is configured (SEARXNG_URL / WEAVIATE_URL), so
    an unconfigured service never registers a dead tool.
    """
    tools: list[BaseTool] = [build_skill_publish_tool()]
    for candidate in (build_search_tool, build_fetch_page_tool, build_kb_search_tool):
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


def _runtime_user_id() -> str:
    """The current run's user_id (runtime context), or 'anonymous'."""
    try:
        rt = get_runtime()
    except Exception:
        return "anonymous"
    ctx = getattr(rt, "context", None) or {}
    if isinstance(ctx, dict):
        return str(ctx.get("user_id") or "anonymous")
    return str(getattr(ctx, "user_id", None) or "anonymous")


class UserShellBackend(LocalShellBackend):
    """LocalShellBackend whose file-tool root and shell cwd resolve per user.

    The single backend of the simple model: every file-tool path and shell
    command resolves to ``WORKSPACE_ROOT/<user_id>/`` — plain host files
    (bind-mounted in compose), isolated per user, and visible to **both**
    the file tools and the `execute` tool. The disk is the source of truth;
    services/workspace only scaffolds the dirs and versions them with git.

    The runtime user comes from the graph execution context (same mechanism
    as `StoreBackend`'s namespace factory); outside a run it is
    "anonymous". `virtual_mode` guards traversal within the user's dir.

    `execution_enabled=False` keeps the `EXECUTE_ENABLED` opt-in: the tool
    stays registered but every command is refused with the standard
    "Execution not available" error.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        inherit_env: bool = False,
        execution_enabled: bool = True,
    ) -> None:
        super().__init__(virtual_mode=True, inherit_env=inherit_env)
        self._execution_enabled = execution_enabled
        self._root = Path(root).resolve()
        # The root is created lazily in _user_dir (never at construction):
        # startup must not fail on an unwritable/missing volume.
        # FilesystemBackend resolves virtual paths under self.cwd; point it
        # at the volume root so per-user dirs stay inside the virtual root
        # (ls displays paths like /alice/... under the workspace route).
        self.cwd = self._root

    def _user_dir(self) -> Path:
        d = self._root / _runtime_user_id()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _resolve_path(self, key: str) -> Path:
        """Resolve a virtual path under the current user's workspace dir."""
        vpath = key if key.startswith("/") else "/" + key
        if ".." in vpath or vpath.startswith("~"):
            raise ValueError("Path traversal not allowed")
        base = self._user_dir()
        full = (base / vpath.lstrip("/")).resolve()
        try:
            full.relative_to(base)
        except ValueError:
            raise ValueError(f"Path:{full} outside root directory: {base}") from None
        return full

    def _to_virtual_path(self, path: Path) -> str:
        """Display paths relative to the USER dir, not the base root.

        The middleware lists a source (e.g. /skills/...) and re-reads the
        returned entry paths; if entries displayed as /<user>/... the
        re-read would resolve under the user dir again and miss.
        """
        return "/" + path.resolve().relative_to(self._user_dir()).as_posix()

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """FilesystemBackend.glob treats None/\"/\" as self.cwd (the volume
        root); the user's dir is the virtual root instead."""
        if path is None or path == "/":
            path = ""
        return super().glob(pattern, path=path)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run the shell command in the current user's workspace dir.

        Refused (standard "Execution not available" error) when the
        `EXECUTE_ENABLED` opt-in is off.
        """
        if not self._execution_enabled:
            return ExecuteResponse(
                output=(
                    "Error: Execution not available. The execute tool is "
                    "disabled (EXECUTE_ENABLED)."
                ),
                exit_code=1,
                truncated=False,
            )
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )
        cwd = self._user_dir()
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)
        try:
            result = subprocess.run(
                command,
                check=False,
                shell=True,  # Intentional: LLM-controlled shell execution
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(cwd),
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout} seconds",
                exit_code=124,
                truncated=False,
            )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)
        output = "\n".join(output_parts) if output_parts else "<no output>"
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True
        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
        return ExecuteResponse(output=output, exit_code=result.returncode, truncated=truncated)


def _global_namespace_factory(_rt: Any) -> tuple[str, ...]:
    """Global namespace shared by all users (agent-level resources)."""
    return GLOBAL_SKILLS_NS


def build_backend(*, store: BaseStore) -> CompositeBackend:
    """Build the agent's filesystem backend: one per-user real workspace.

    Every path (file tools AND the `execute` tool) resolves to
    ``WORKSPACE_ROOT/<user_id>/`` via `UserShellBackend` — plain host files
    (bind-mounted in compose, so user dirs show up on the host), isolated
    per user, no virtual mounts. Durability and versioning come from
    `services/workspace` (git auto-commit after each run).

    The `store` argument is kept for API stability (the resources API and
    tests pass it); the store itself is not consulted at build time.

    `EXECUTE_ENABLED=false` keeps the same backend but refuses every
    execute command (opt-in shell).
    """
    return CompositeBackend(
        default=UserShellBackend(
            root=settings.workspace_root,
            inherit_env=runtime_settings.execute_inherit_env(),
            execution_enabled=runtime_settings.execute_enabled(),
        ),
        routes={},
    )


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
    memory: list[str] | None = None,
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
        memory: MemoryMiddleware sources — AGENTS.md files auto-loaded into
            the system prompt before every run. None = the default
            "/memories/AGENTS.md" (per-user via the backend); [] = no
            memory middleware.
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
        memory=memory if memory is not None else [MEMORY_SOURCE],
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
        mcp_provider: Callable[[str], Awaitable[tuple[list[BaseTool], dict[str, list[str]]]]]
        | None = None,
    ) -> None:
        """Args:
        checkpointer/store/backend: shared persistence + filesystem backend
            (one instance for every graph).
        mcp_tools: static tools loaded from MCP servers (tests only — use
            `mcp_provider` for per-user MCP).
        extra_tools: additional tools, e.g. the SearXNG web_search tool.
        tools_by_server: tool name -> MCP server name attribution, for
            per-agent tool selection (`web_search` is a built-in pseudo-tool).
        model_factory: test hook returning a chat model for (model, temperature).
        static_default: when set (tests), every resolve() returns this graph.
        mcp_provider: async (username) -> (tools, tools_by_server); resolves
            the user's own MCP servers at graph-build time.
        """
        self._checkpointer = checkpointer
        self._store = store
        self._backend = backend
        self._mcp_tools = list(mcp_tools or [])
        self._extra_tools = list(extra_tools or [])
        self._tools_by_server = dict(tools_by_server or {})
        self._mcp_provider = mcp_provider
        self._model_factory = model_factory
        self._static_default = static_default
        self._cache: OrderedDict[str, CompiledStateGraph] = OrderedDict()
        self._max_cache = max_cache

    @property
    def backend(self) -> CompositeBackend:
        """Current filesystem backend (per-user workspace)."""
        return self._backend

    async def resolve(
        self, name: str = "default", username: str = "anonymous"
    ) -> CompiledStateGraph:
        """The compiled graph for an agent config (user -> global -> builtin default).

        The graph is per-user: the user's own llm connection (BYOK model
        credentials) and their own MCP tools are baked in at build time.
        Raises KeyError when the config does not exist.
        """
        if self._static_default is not None:
            return self._static_default
        spec = await load_spec(self._store, name, username)
        if spec is None:
            raise KeyError(name)
        return await self._build_for(spec, username)

    async def _build_for(self, spec: AgentSpec, username: str) -> CompiledStateGraph:
        key = f"{username}:{spec.fingerprint()}"
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        tools = await self._select_tools(spec, username)
        model = await self._resolve_model(spec)
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

    async def _mcp_for(self, username: str) -> tuple[list[BaseTool], dict[str, list[str]]]:
        """The user's MCP tools: per-user provider when wired, else static lists."""
        if self._mcp_provider is not None:
            return await self._mcp_provider(username)
        return list(self._mcp_tools), dict(self._tools_by_server)

    async def _select_tools(self, spec: AgentSpec, username: str) -> list[BaseTool]:
        """Tools for the agent: inherited (all) or the named selection."""
        if spec.tools is None:
            mcp_tools, _ = await self._mcp_for(username)
            return list(mcp_tools) + list(self._extra_tools)
        mcp_tools, tools_by_server = await self._mcp_for(username)
        selected: list[BaseTool] = []
        by_name = {t.name: t for t in mcp_tools}
        for name in spec.tools:
            if name in BUILTIN_PSEUDO_TOOLS:
                selected.extend(t for t in self._extra_tools if t.name == name)
                continue
            for tool_name in tools_by_server.get(name, []):
                if tool_name in by_name and by_name[tool_name] not in selected:
                    selected.append(by_name[tool_name])
        return selected

    async def _resolve_model(self, spec: AgentSpec) -> str | BaseChatModel:
        """The chat model for an agent spec (per-agent connection binding).

        When the spec names a saved `connection`, its base_url + api_token
        route the completions (so a model picked from any source in
        `GET /connections/models` works); otherwise the default llm
        connection is used. There is no env fallback — without a saved
        connection the agent refuses to build (never silently env-defaults).
        """
        if self._model_factory is not None:
            return self._model_factory(spec.model, spec.temperature)
        conn = None
        if spec.connection:
            conn = await persistence.connections.get(spec.connection)
            if conn is None or conn.get("kind") != "llm":
                raise ValueError(
                    f"Agent '{spec.name}' references unknown llm connection "
                    f"'{spec.connection}' — save it via POST /connections first"
                )
        kwargs = model_kwargs_from(conn) if conn is not None else llm_model_kwargs()
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.thinking is not None:
            kwargs["reasoning_effort"] = spec.thinking
        # Model name: the spec's explicit model wins; otherwise the bound
        # connection's extra.model; otherwise nothing is configured and the
        # agent refuses to build (no silent default model).
        model_name = (
            spec.model or ((conn or {}).get("extra") or {}).get("model") or llm_model_name()
        )
        if model_name is None:
            raise ValueError(
                "No model configured: save a default llm connection carrying "
                "the model, e.g. POST /connections "
                '{"name": "zen", "kind": "llm", "base_url": "https://.../v1", '
                '"api_token": "sk-...", "extra": {"model": '
                '"openai:deepseek-v4-flash"}, "is_default": true}'
            )
        if not kwargs:
            raise ValueError(
                "No default 'llm' connection configured — create one via "
                "POST /connections (kind=llm, is_default=true, base_url + "
                "api_token); env credentials are never used."
            )
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model_name, **kwargs)
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
        return await self._resolve_model(spec)

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
        self._backend = build_backend(store=store)
        self._cache.clear()

    def invalidate(self) -> None:
        """Drop cached graphs; the next request rebuilds from current resources."""
        self._cache.clear()

    def set_static_default(self, graph: CompiledStateGraph | None) -> None:
        """Test hook: every resolve() returns this graph (or None to disable).

        Replaces the static default passed at construction (e.g. when the
        graph must be built with persistence instances that only exist after
        startup). Chats resolve through the registry, so tests swapping only
        `app.state.agent` must also call this for the chat path.
        """
        self._static_default = graph
        self._cache.clear()
