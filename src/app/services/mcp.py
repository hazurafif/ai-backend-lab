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

MCP resources are a client-side concept, so the LangChain agent cannot see
them — `MCPServers.list_resources` / `read_resource` wrap the raw MCP SDK
(resource templates included, which list_resources alone hides) and
`connect()` registers them as the `mcp_list_resources` / `mcp_read_resource`
bridge tools, attributed to every connected server.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from contextlib import suppress
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session
from langgraph.store.base import BaseStore
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import (
    METHOD_NOT_FOUND,
    BlobResourceContents,
    CallToolResult,
    TextContent,
    TextResourceContents,
)
from pydantic import BaseModel, Field

from ..core.config import settings
from ..schema.mcp_schema import McpResourceOut, McpResourceReadOut, McpToolCallOut
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


# MCP spec error code for "resource not found" (-32002) — the SDK does not
# export it. METHOD_NOT_FOUND (-32601) means the server never implements the
# resources feature (e.g. context7-style tool-only servers) and is treated as
# "try the next server" in fan-out, never as a failure.
RESOURCE_NOT_FOUND_CODE = -32002

_RESOURCE_NOT_FOUND_RE = re.compile(
    r"unknown resource|resource not found|not found: |resource [a-z0-9_.:/-]+ not found",
    re.IGNORECASE,
)


def _is_resource_not_found(error: McpError) -> bool:
    """True when an MCP error means "this server doesn't have that resource".

    Covers the spec's -32002 code, FastMCP/gofastmcp message dialects, and
    -32601 (server without a resources feature at all).
    """
    return error.error.code in (RESOURCE_NOT_FOUND_CODE, METHOD_NOT_FOUND) or bool(
        _RESOURCE_NOT_FOUND_RE.search(str(error))
    )


def _result_text(result: CallToolResult) -> str:
    """Concatenated text of a result's text content blocks (error messages)."""
    return " ".join(c.text for c in result.content if isinstance(c, TextContent) and c.text)


class McpToolNotFoundError(Exception):
    """No configured server knows the requested tool (maps to HTTP 404)."""


class McpResourceNotFoundError(McpToolNotFoundError):
    """No configured server has the requested resource (maps to HTTP 404)."""


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
        resource_tools = build_resource_tools(self)
        if resource_tools and self.tools_by_server:
            self.tools.extend(resource_tools)
            for names in self.tools_by_server.values():
                names.extend(t.name for t in resource_tools)
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

    # ------------------------------------------------------------------
    # resources: raw-SDK list (static + templates) and read
    # ------------------------------------------------------------------

    async def list_resources(self, server_hint: str | None = None) -> list[McpResourceOut]:
        """List resources + resource templates from MCP server(s).

        `server_hint` restricts to one server; otherwise every configured
        server contributes. Servers that fail are recorded in `self.failed`
        and skipped (one dead server never kills the listing); servers that
        don't implement the resources feature (-32601) are skipped quietly.
        Templates are included because `list_resources()` hides them —
        parameterized resources are only discoverable via
        `list_resource_templates()`.
        """
        candidates = [server_hint] if server_hint is not None else list(self._config)
        if server_hint is not None and server_hint not in self._config:
            raise McpToolNotFoundError(f"Unknown MCP server {server_hint!r}")

        items: list[McpResourceOut] = []
        for server in candidates:
            try:
                session = await self._get_session(server)
                listed = await session.list_resources()
                templates = await session.list_resource_templates()
            except McpError as e:
                if e.error.code == METHOD_NOT_FOUND:
                    continue  # tool-only server: no resources feature
                await self._close_session(server)
                self.failed[server] = str(e)
                logger.warning("MCP resource listing failed: server=%s error=%s", server, e)
                continue
            except Exception as e:
                await self._close_session(server)
                self.failed[server] = str(e)
                logger.warning("MCP resource listing failed: server=%s error=%s", server, e)
                continue
            for r in listed.resources:
                items.append(
                    McpResourceOut(
                        server=server,
                        uri=str(r.uri),
                        name=r.name,
                        description=r.description,
                        mime_type=r.mimeType,
                    )
                )
            for t in templates.resourceTemplates:
                items.append(
                    McpResourceOut(
                        server=server,
                        uri_template=t.uriTemplate,
                        name=t.name,
                        description=t.description,
                        mime_type=t.mimeType,
                    )
                )
        return items

    async def read_resource(self, uri: str, server_hint: str | None = None) -> McpResourceReadOut:
        """Read a resource from an MCP server (template instantiations included).

        Routing mirrors `call_tool`: `server_hint` is authoritative (no
        fallback); without it the servers are tried in config order and the
        first one that has the resource wins. A server without the resource
        (-32002) or without the resources feature at all (-32601) is skipped;
        any other failure is a transport error and the cached session is
        evicted for reconnect. Raises `McpResourceNotFoundError` when no
        server matched, `McpTransportError` on protocol/transport failures.
        """
        candidates = [server_hint] if server_hint is not None else list(self._config)
        if server_hint is not None and server_hint not in self._config:
            raise McpToolNotFoundError(f"Unknown MCP server {server_hint!r}")

        not_found: list[str] = []
        for server in candidates:
            try:
                session = await self._get_session(server)
                result = await session.read_resource(uri)
            except McpError as e:
                if _is_resource_not_found(e):
                    not_found.append(f"{server}: {e}")
                    continue
                await self._close_session(server)
                logger.warning(
                    "MCP resource read failed: server=%s uri=%s error=%s", server, uri, e
                )
                raise McpTransportError(f"{server}: {e}") from e
            except Exception as e:
                await self._close_session(server)
                logger.warning(
                    "MCP resource read failed: server=%s uri=%s error=%s", server, uri, e
                )
                raise McpTransportError(f"{server}: {e}") from e
            blocks: list[dict[str, Any]] = []
            for content in result.contents:
                if isinstance(content, TextResourceContents):
                    blocks.append({"type": "text", "text": content.text})
                elif isinstance(content, BlobResourceContents):
                    blocks.append(
                        {
                            "type": "blob",
                            "blob": content.blob,
                            "mime_type": content.mimeType,
                        }
                    )
            logger.info("MCP resource read ok: server=%s uri=%s", server, uri)
            return McpResourceReadOut(uri=uri, content=blocks)
        detail = f" (tried: {', '.join(not_found)})" if not_found else ""
        raise McpResourceNotFoundError(f"Resource {uri!r} not found on any MCP server{detail}")

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


class ListResourcesInput(BaseModel):
    """Arguments for the `mcp_list_resources` tool."""

    server: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional MCP server name to list resources from (default: every configured server)."
        ),
    )


class ReadResourceInput(BaseModel):
    """Arguments for the `mcp_read_resource` tool."""

    uri: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description=(
            "Resource URI to read. For templated resources, substitute "
            "concrete values first (e.g. 'weather://paris/current' for the "
            "template 'weather://{city}/current')."
        ),
    )
    server: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional MCP server name to read from (default: try every configured server in order)."
        ),
    )


def _format_resource_list(items: list[McpResourceOut]) -> str:
    """Human-readable listing of resources/templates, grouped by server."""
    if not items:
        return "No MCP resources found."
    lines: list[str] = []
    seen_servers: set[str] = set()
    for item in items:
        if item.server not in seen_servers:
            seen_servers.add(item.server)
            lines.append(f"MCP server '{item.server}':")
        if item.uri_template is not None:
            lines.append(
                f"  template {item.uri_template} - {item.description or item.name or ''}".rstrip(
                    " -"
                )
            )
        else:
            lines.append(f"  {item.uri} - {item.description or item.name or ''}".rstrip(" -"))
    return "\n".join(lines)


def build_resource_tools(servers: MCPServers) -> list[BaseTool]:
    """Agent bridge tools for MCP resources, bound to `servers`.

    The MCP SDK gives clients list/read methods, but the agent can only call
    what is in its `tools=` list — so the two methods are wrapped as LangChain
    tools: `mcp_list_resources` (discovery: static resources AND RFC 6570
    templates, which list_resources alone hides) and `mcp_read_resource`
    (read any URI, template instantiations included).
    """

    async def list_resources(server: str | None = None) -> str:
        """List what MCP resources are available to read."""
        try:
            items = await servers.list_resources(server_hint=server)
        except McpToolNotFoundError as exc:
            return f"Error: {exc}"
        text = _format_resource_list(items)
        if server is None and servers.failed:
            text += "\n\nUnreachable servers: " + ", ".join(
                f"{name} ({reason})" for name, reason in servers.failed.items()
            )
        return text

    async def read_resource(uri: str, server: str | None = None) -> str:
        """Read an MCP resource by URI (template instantiations included)."""
        try:
            out = await servers.read_resource(uri, server_hint=server)
        except (McpResourceNotFoundError, McpTransportError, McpToolNotFoundError) as exc:
            return f"Error: {exc}"
        parts: list[str] = []
        for block in out.content:
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "blob":
                parts.append(
                    f"[binary content: mime={block.get('mime_type') or 'application/octet-stream'}, "
                    f"base64 length={len(str(block.get('blob', '')))}]"
                )
        return "\n".join(parts) if parts else "(empty resource)"

    return [
        StructuredTool.from_function(
            coroutine=list_resources,
            name="mcp_list_resources",
            description=(
                "List resources and resource templates exposed by the configured "
                "MCP servers. Use this to discover data sources before reading "
                "one. Resource templates contain {placeholders} (e.g. "
                "'weather://{city}/current') — substitute concrete values to "
                "form a URI, then pass it to mcp_read_resource."
            ),
            args_schema=ListResourcesInput,
        ),
        StructuredTool.from_function(
            coroutine=read_resource,
            name="mcp_read_resource",
            description=(
                "Read a resource from an MCP server by URI. For templated "
                "resources, instantiate the template with concrete values "
                "first (see mcp_list_resources), e.g. 'weather://paris/current' "
                "for 'weather://{city}/current'. Text content is returned "
                "verbatim; binary content is summarized with its MIME type "
                "and size."
            ),
            args_schema=ReadResourceInput,
        ),
    ]


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
