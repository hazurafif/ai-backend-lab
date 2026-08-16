"""MCP wire models: app tools proxy (POST /mcp/tools/call) + resources.

The tools proxy format mirrors the MCP `CallToolResult` so the response can
be handed back to the renderer verbatim. Tool names are the hashed FastMCP
app-tool form (`<12 lowercase hex>_<name>`, e.g. `10c0803009ff_save_contact`)
or a plain MCP tool name — the backend passes them through without ever
needing to know the mapping (FastMCP resolves the hash server-side).

The resource models carry the raw-SDK `resources/list` + `resources/read`
results (templates included) for the agent bridge tools.
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


class McpResourceOut(BaseModel):
    """A resource or resource template advertised by an MCP server."""

    server: str = Field(
        ...,
        max_length=64,
        description="Name of the MCP server exposing the resource",
    )
    uri: str | None = Field(
        default=None,
        description="Concrete resource URI (static resources).",
    )
    uri_template: str | None = Field(
        default=None,
        description=(
            "RFC 6570 URI template (dynamic resources), e.g. "
            "'weather://{city}/current'; substitute concrete values to read."
        ),
    )
    name: str | None = Field(default=None, description="Human-readable name")
    description: str | None = Field(default=None, description="What the resource provides")
    mime_type: str | None = Field(default=None, description="MIME type of the resource content")


class McpResourceReadOut(BaseModel):
    """Result of reading an MCP resource (ReadResourceResult passthrough)."""

    uri: str = Field(..., description="Resource URI that was read")
    content: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Content blocks: [{'type': 'text', 'text': '...'}] for text, "
            "[{'type': 'blob', 'blob': '<base64>', 'mime_type': '...'}] for binary"
        ),
    )
