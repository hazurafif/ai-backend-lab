"""MCP client: connects the agent to MCP servers (e.g. built with gofastmcp).

Server config is a dict of name -> connection, **per user**: each user's
servers live in their own store namespace (`("user", "mcp_servers",
<username>)`, managed via the per-user /mcp/servers CRUD and /agent/tools
APIs). Nothing is shared between users — fresh, private config per user.

Config shape:

    {
      "weather": {
        "url": "http://localhost:8090/mcp",
        "transport": "streamable_http",
        "headers": {"Authorization": "Bearer ..."}
      },
      "local-tool": {
        "command": "gofastmcp-tool",
        "args": ["serve"],
        "transport": "stdio",
        "env": {"FOO": "bar"}
      }
    }

- streamable_http: for gofastmcp (Go) servers deployed as web services
- stdio: for gofastmcp binaries run as subprocesses

Tools are fetched lazily per user (first agent build / tools call / explicit
reconnect) and passed to `create_deep_agent(tools=...)` via the registry's
per-user MCP provider.

`MCPServers.call_tool` is the MCP apps tools proxy (POST /mcp/tools/call):
prefab renderers forward `tools/call` messages here and the backend invokes
the tool over a cached, multiplexed MCP session (one per server, stdio
included). Tool names arrive in FastMCP's hashed app-tool form
(`<12 lowercase hex>_<name>`) and are passed through verbatim — FastMCP
resolves the hash server-side via `get_tool_by_hash`, so the proxy never
needs to know the mapping. With no `server_hint` the call fans out across
the user's configured servers; a server that doesn't know the name reports
tool-not-found and the next server is tried (first hit wins).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from contextlib import suppress
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session
from langgraph.store.base import BaseStore
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

from ..core.config import settings
from ..schema.mcp_schema import McpToolCallOut
from .resources import load_tool_server_configs

logger = logging.getLogger(__name__)

# Tool-not-found signals across MCP servers: gofastmcp/mcp-go raises a
# JSON-RPC error "tool not found", current FastMCP "Not found: '<name>'",
# older FastMCP returns an isError result with "Unknown tool: <name>", and
# non-FastMCP servers (e.g. context7) answer "Tool <name> not found".
_TOOL_NOT_FOUND_RE = re.compile(
    r"unknown tool|tool not found|not found: |tool [a-z0-9_.-]+ not found",
    re.IGNORECASE,
)


def _is_tool_not_found(message: str) -> bool:
    return bool(_TOOL_NOT_FOUND_RE.search(message))


def _result_text(result: CallToolResult) -> str:
    """Concatenated text of a result's text content blocks (error messages)."""
    return " ".join(c.text for c in result.content if isinstance(c, TextContent) and c.text)


class McpToolNotFoundError(Exception):
    """No configured server knows the requested tool (maps to HTTP 404)."""


class McpTransportError(Exception):
    """An MCP server failed at the transport/protocol level (maps to HTTP 502)."""


class MCPServers:
    def __init__(self) -> None:
        self._config: dict[str, dict[str, Any]] = {}
        self._client: MultiServerMCPClient | None = None
        self._connected = False
        # server name -> cached, multiplexed ClientSession (lazy, proxy only)
        self._sessions: dict[str, ClientSession] = {}
        # Per-server owner tasks + close events: anyio cancel scopes are
        # task-local, so a session's context managers must be entered and
        # exited in the same task (a session created in a request task cannot
        # be torn down from another task or out of LIFO order). Each owner
        # task enters create_session() and parks until its close event fires.
        self._owners: dict[str, asyncio.Task] = {}
        self._close_events: dict[str, asyncio.Event] = {}
        self._session_lock = asyncio.Lock()
        self.tools: list[BaseTool] = []
        self.failed: dict[str, str] = {}
        # server name -> tool names it exposes (per-agent tool selection)
        self.tools_by_server: dict[str, list[str]] = {}

    @property
    def connected(self) -> bool:
        """Whether connect() ran at least once (configs may still be empty)."""
        return self._connected

    @property
    def names(self) -> list[str]:
        return list(self._config)

    async def connect(
        self, store: BaseStore | None = None, *, username: str = "anonymous"
    ) -> list[BaseTool]:
        """Connect to the user's configured MCP servers and fetch their tools.

        Config comes from the user's own store namespace (per-user CRUD API),
        never from other users or env. Re-connecting replaces the previous
        configuration. Tools are fetched per server so `tools_by_server` can
        attribute tool names to their server (tool names themselves are not
        prefixed). Wrapper keys (e.g. the CRUD API's `enabled` flag) are
        stripped so configs can be handed straight to create_session.
        """
        if store is not None:
            raw_config = await load_tool_server_configs(store, username)
        else:
            # Standalone use (scripts, tests, ad-hoc clients): env/file config.
            raw_config = settings.load_mcp_servers()
        self._connected = True
        self._config = {
            name: {k: v for k, v in cfg.items() if k != "enabled"}
            for name, cfg in raw_config.items()
        }
        # Proxy sessions are per-server and outlive connects only by accident;
        # drop any cached ones from a previous configuration.
        await self._close_all_sessions()
        if not self._config:
            logger.info("No MCP servers configured")
            self._client = None
            return []

        client = MultiServerMCPClient(self._config)
        self._client = client
        self.tools = []
        self.tools_by_server = {}
        self.failed = {}
        seen: set[str] = set()
        for server_name in self._config:
            try:
                server_tools = await client.get_tools(server_name=server_name)
            except Exception as exc:  # one dead server must not kill the rest
                self.failed[server_name] = str(exc)
                logger.warning("MCP server %s unreachable, skipping: %s", server_name, exc)
                continue
            names: list[str] = []
            for tool in server_tools:
                if tool.name in seen:
                    continue  # duplicate tool name across servers: first wins
                seen.add(tool.name)
                names.append(tool.name)
                self.tools.append(tool)
            self.tools_by_server[server_name] = names
        if self.failed:
            logger.info(
                "Connected MCP servers: %s (%d tools total); failed: %s",
                self.names,
                len(self.tools),
                self.failed,
            )
        else:
            logger.info("Connected MCP servers: %s (%d tools total)", self.names, len(self.tools))
        return self.tools

    # ------------------------------------------------------------------
    # tools/call proxy (POST /mcp/tools/call)
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        server_hint: str | None = None,
    ) -> McpToolCallOut:
        """Invoke a tool on an MCP server; passthrough of the CallToolResult.

        Routing: `server_hint` first (no fallback — the renderer knows which
        server an app came from); otherwise fan out across configured servers
        in config order, treating tool-not-found as "try the next server".
        First hit wins: a server that knows the name is authoritative, and
        its result (success or `isError`) is returned as-is. Raises
        `McpToolNotFoundError` when no server matched, `McpTransportError`
        on transport/protocol failures.
        """
        candidates = [server_hint] if server_hint is not None else list(self._config)
        if server_hint is not None and server_hint not in self._config:
            raise McpToolNotFoundError(f"Unknown MCP server {server_hint!r}")

        timeout = timedelta(seconds=settings.mcp_tool_call_timeout)
        not_found: list[str] = []
        for server in candidates:
            try:
                session = await self._get_session(server)
                result = await session.call_tool(
                    name, arguments or {}, read_timeout_seconds=timeout
                )
            except McpError as e:
                if _is_tool_not_found(str(e)):
                    not_found.append(f"{server}: {e}")
                    continue
                # Protocol-level failure (dead session, invalid params, ...):
                # drop the cached session so the next call reconnects, then
                # surface the upstream error as a 502.
                await self._close_session(server)
                logger.warning("MCP tool call failed: server=%s tool=%s error=%s", server, name, e)
                raise McpTransportError(f"{server}: {e}") from e
            except Exception as e:
                # Transport-level failure: drop the cached session so the
                # next call reconnects (stdio subprocesses included).
                await self._close_session(server)
                logger.warning("MCP tool call failed: server=%s tool=%s error=%s", server, name, e)
                raise McpTransportError(f"{server}: {e}") from e
            if result.isError and _is_tool_not_found(_result_text(result)):
                not_found.append(f"{server}: {_result_text(result)}")
                continue
            logger.info(
                "MCP tool call ok: server=%s tool=%s is_error=%s", server, name, result.isError
            )
            return McpToolCallOut(
                content=[
                    block.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for block in result.content
                ],
                structuredContent=result.structuredContent,
                isError=result.isError,
            )
        detail = f" (tried: {', '.join(not_found)})" if not_found else ""
        raise McpToolNotFoundError(f"Tool {name!r} not found on any MCP server{detail}")

    async def _get_session(self, server_name: str) -> ClientSession:
        """Cached, multiplexed ClientSession for a server (lazy connect).

        One session per server, shared across concurrent calls (MCP request
        ids correlate responses). Stdio servers stay running as a single
        subprocess until the session is dropped or the app shuts down.

        The session is owned by a dedicated task (see __init__): anyio cancel
        scopes are task-local, so entering the transport context managers in
        the request task and exiting them later from another task (or out of
        LIFO order) breaks them. The owner task parks on a close event; all
        teardown happens inside it.
        """
        session = self._sessions.get(server_name)
        if session is not None:
            return session
        async with self._session_lock:
            session = self._sessions.get(server_name)
            if session is not None:
                return session
            connection = self._config.get(server_name)
            if connection is None:
                raise McpToolNotFoundError(f"Unknown MCP server {server_name!r}")
            event = asyncio.Event()
            self._close_events[server_name] = event
            ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
            task = asyncio.create_task(self._session_owner(server_name, connection, ready, event))
            self._owners[server_name] = task
            try:
                session = await ready
            except BaseException:
                if not task.done():
                    await self._close_session(server_name)
                raise
            self._sessions[server_name] = session
            return session

    async def _session_owner(
        self,
        server_name: str,
        connection: dict[str, Any],
        ready: asyncio.Future[ClientSession],
        close_event: asyncio.Event,
    ) -> None:
        """Owner task: enter the session's context managers and park until close.

        Runs entirely in one task, so the stdio/HTTP transport cancel scopes
        are entered and exited here (never in request tasks). Reports the
        connected session through `ready`; exits the context managers (and
        the stdio subprocess) when `close_event` fires.
        """
        cm = create_session(connection)
        try:
            session = await cm.__aenter__()
            await session.initialize()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
            with suppress(Exception):
                await cm.__aexit__(*sys.exc_info())
            return
        if not ready.done():
            ready.set_result(session)
        logger.info(
            "MCP session connected: server=%s transport=%s",
            server_name,
            connection.get("transport"),
        )
        try:
            await close_event.wait()
        finally:
            await cm.__aexit__(None, None, None)
            logger.info("MCP session closed: server=%s", server_name)

    async def _close_session(self, server_name: str) -> None:
        """Signal a server's owner task to tear down its session (best effort)."""
        self._sessions.pop(server_name, None)
        event = self._close_events.pop(server_name, None)
        task = self._owners.pop(server_name, None)
        if event is not None:
            event.set()
        if task is not None:
            with suppress(Exception):
                await task

    async def _close_all_sessions(self) -> None:
        for name in list(self._sessions):
            await self._close_session(name)

    async def close(self) -> None:
        await self._close_all_sessions()
        self.tools = []
        self.tools_by_server = {}
        self._client = None
        self._config = {}
        self._connected = False


class MCPRegistry:
    """Per-user MCPServers instances, lazily connected and cached.

    Each user's MCP servers are a separate `MCPServers` instance (their own
    config, sessions, tools). Instances are created on first use and dropped
    by `invalidate(username)` after CRUD mutations, so the next access
    reconnects from the user's current config.
    """

    def __init__(self) -> None:
        self._instances: dict[str, MCPServers] = {}
        self._lock = asyncio.Lock()

    async def get(self, username: str, store: BaseStore | None = None) -> MCPServers:
        """The user's MCPServers instance (created on first use).

        When `store` is given and the instance was never connected, connects
        it from the user's stored config first.
        """
        instance = self._instances.get(username)
        if instance is None:
            async with self._lock:
                instance = self._instances.get(username)
                if instance is None:
                    instance = MCPServers()
                    self._instances[username] = instance
        if store is not None and not instance.connected:
            await instance.connect(store=store, username=username)
        return instance

    async def connect(self, username: str, store: BaseStore) -> MCPServers:
        """(Re)connect the user's MCP servers from their stored config (reconnect)."""
        instance = await self.get(username, store=None)
        await instance.connect(store=store, username=username)
        return instance

    async def invalidate(self, username: str) -> None:
        """Close and drop the user's MCP instance (after CRUD mutations)."""
        instance = self._instances.pop(username, None)
        if instance is not None:
            with suppress(Exception):
                await instance.close()

    async def close(self) -> None:
        """Close every user's MCP instance (app shutdown)."""
        for instance in list(self._instances.values()):
            with suppress(Exception):
                await instance.close()
        self._instances.clear()


mcp_servers = MCPRegistry()
