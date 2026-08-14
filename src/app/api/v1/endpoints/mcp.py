"""MCP apps tools proxy: forwards prefab renderer `tools/call` to MCP servers.

One endpoint, same trust level as /api/chat (any authenticated user):

    POST /mcp/tools/call
    { "name": "<hash>_save_contact", "arguments": {...}, "server_hint": "contacts" }
    -> { "content": [...], "structuredContent": ..., "isError": false }

Routing happens in `services/mcp.MCPServers.call_tool`: `server_hint` first,
otherwise fan-out across configured servers (first hit wins). The tool name
is passed through verbatim — FastMCP resolves the `<hash>_<name>` mapping
server-side. Upstream `isError: true` results pass through with HTTP 200
(the renderer handles them); transport failures map to 502 and no-match to
404. Auth headers never leave the backend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ....core.dependencies import get_current_user
from ....core.exceptions import BadGateway, NotFound
from ....schema.mcp_schema import McpToolCallIn, McpToolCallOut
from ....services.mcp import McpToolNotFoundError, McpTransportError, mcp_servers

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.post("/tools/call", response_model=McpToolCallOut)
async def call_tool(body: McpToolCallIn, _: dict = Depends(get_current_user)):
    """Invoke a tool on a configured MCP server (prefab app button → server tool)."""
    try:
        return await mcp_servers.call_tool(body.name, body.arguments, server_hint=body.server_hint)
    except McpToolNotFoundError as e:
        raise NotFound(str(e)) from None
    except McpTransportError as e:
        raise BadGateway(str(e)) from None
