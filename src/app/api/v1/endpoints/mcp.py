"""MCP routes: per-user tool server CRUD + the apps tools proxy.

Each user brings their own MCP servers (per-user store namespace, see
`services/resources`): CRUD at /mcp/servers and the tools/call proxy route to
the caller's own MCPServers instance — no user can invoke another user's
servers. The tools proxy has the same trust level as /api/chat (any
authenticated user):

    POST /mcp/tools/call
    { "name": "<hash>_save_contact", "arguments": {...}, "server_hint": "contacts" }
    -> { "content": [...], "structuredContent": ..., "isError": false }

Routing happens in `services/mcp.MCPServers.call_tool`: `server_hint` first,
otherwise fan-out across the user's configured servers (first hit wins). The
tool name is passed through verbatim — FastMCP resolves the `<hash>_<name>`
mapping server-side. Upstream `isError: true` results pass through with HTTP
200 (the renderer handles them); transport failures map to 502 and no-match
to 404. Auth headers never leave the backend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....core.exceptions import BadGateway, Conflict, NotFound
from ....schema.agent_schema import ToolServerIn, ToolServerOut
from ....schema.mcp_schema import McpToolCallIn, McpToolCallOut
from ....services import resources
from ....services.mcp import McpToolNotFoundError, McpTransportError, mcp_servers

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.post("/tools/call", response_model=McpToolCallOut)
async def call_tool(body: McpToolCallIn, current_user: dict = Depends(get_current_user)):
    """Invoke a tool on one of the caller's configured MCP servers."""
    instance = await mcp_servers.get(current_user["username"], persistence.store)
    try:
        return await instance.call_tool(body.name, body.arguments, server_hint=body.server_hint)
    except McpToolNotFoundError as e:
        raise NotFound(str(e)) from None
    except McpTransportError as e:
        raise BadGateway(str(e)) from None


# ---------------------------------------------------------------------------
# per-user MCP tool server CRUD (each user manages their own connections)
# ---------------------------------------------------------------------------


@router.get("/servers", response_model=list[ToolServerOut])
async def list_servers(current_user: dict = Depends(get_current_user)):
    """The caller's MCP tool servers, by name."""
    return await resources.list_tool_servers(persistence.store, current_user["username"])


@router.post("/servers", response_model=ToolServerOut, status_code=201)
async def create_server(
    body: ToolServerIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    if await resources.get_tool_server(persistence.store, current_user["username"], body.name):
        raise Conflict(f"Tool server '{body.name}' already exists")
    out = await resources.create_tool_server(persistence.store, current_user["username"], body)
    await mcp_servers.invalidate(current_user["username"])
    request.app.state.agents.invalidate()
    return out


@router.get("/servers/{name}", response_model=ToolServerOut)
async def get_server(name: str, current_user: dict = Depends(get_current_user)):
    server = await resources.get_tool_server(persistence.store, current_user["username"], name)
    if server is None:
        raise NotFound("Tool server not found")
    return server


@router.put("/servers/{name}", response_model=ToolServerOut)
async def update_server(
    name: str,
    body: ToolServerIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    try:
        out = await resources.update_tool_server(
            persistence.store, current_user["username"], name, body
        )
    except KeyError:
        raise NotFound("Tool server not found") from None
    await mcp_servers.invalidate(current_user["username"])
    request.app.state.agents.invalidate()
    return out


@router.delete("/servers/{name}", status_code=204)
async def delete_server(
    name: str, request: Request, current_user: dict = Depends(get_current_user)
):
    if not await resources.delete_tool_server(persistence.store, current_user["username"], name):
        raise NotFound("Tool server not found")
    await mcp_servers.invalidate(current_user["username"])
    request.app.state.agents.invalidate()


@router.post("/servers/reconnect")
async def reconnect_servers(request: Request, current_user: dict = Depends(get_current_user)):
    """Reconnect the caller's MCP servers from their stored config (live).

    Unreachable servers are recorded per-server (not fatal) so the healthy
    ones still load; the response reports them under `failed`. Cached agent
    graphs are dropped so the next run picks up the new tool set.
    """
    instance = await mcp_servers.connect(current_user["username"], persistence.store)
    request.app.state.agents.invalidate()
    return {
        "connected": [n for n in instance.names if n not in instance.failed],
        "tools": len(instance.tools),
        "failed": instance.failed,
    }
