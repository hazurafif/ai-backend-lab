"""MCP app tools proxy models (POST /mcp/tools/call).

Wire format mirrors the MCP `CallToolResult` so the response can be handed
back to the renderer verbatim. Tool names are the hashed FastMCP app-tool
form (`<12 lowercase hex>_<name>`, e.g. `10c0803009ff_save_contact`) or a
plain MCP tool name — the backend passes them through without ever needing
to know the mapping (FastMCP resolves the hash server-side).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# MCP identifier charset (plus '.' for namespaced tools), capped at the MCP
# spec's 128-char tool-name limit. Covers hashed app-tool names and plain
# names; keeps the proxy a passthrough, not a name interpreter.
TOOL_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"


class McpToolCallIn(BaseModel):
    """Invoke a tool on one of the configured MCP servers."""

    name: str = Field(
        ...,
        pattern=TOOL_NAME_PATTERN,
        min_length=1,
        max_length=128,
        description=(
            "Tool name as seen on the wire: FastMCP hashed app-tool form "
            "'<hash>_<name>' or a plain MCP tool name"
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments (validated by the MCP server against the tool schema)",
    )
    server_hint: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional server name to route to (skips fan-out). When omitted the "
            "backend tries each configured server in order and first hit wins."
        ),
    )


class McpToolCallOut(BaseModel):
    """Passthrough of the MCP CallToolResult (same JSON shape)."""

    content: list[dict[str, Any]] = Field(
        ..., description='Content blocks, e.g. [{"type": "text", "text": "Saved"}]'
    )
    structuredContent: dict[str, Any] | None = Field(
        default=None, description="Optional structured result of the tool call"
    )
    isError: bool = Field(
        default=False,
        description="True when the server reported a tool execution error (still HTTP 200)",
    )
