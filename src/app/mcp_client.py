"""MCP client: connects the agent to MCP servers (e.g. built with gofastmcp).

Server config is a dict of name -> connection. Sources, in priority order:

1. The durable store namespace ("agent", "mcp_servers") — managed via the
   /agent/tools CRUD API (Postgres-backed in production).
2. `MCP_SERVERS_JSON` env var or `mcp_servers.json` (see the example file).

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

Tools are fetched once at startup (or via POST /agent/tools/reconnect) and
passed to `create_deep_agent(tools=...)`.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.store.base import BaseStore

from .agent_resources import load_tool_server_configs
from .config import settings

logger = logging.getLogger(__name__)


class MCPServers:
    def __init__(self) -> None:
        self._config: dict[str, dict[str, Any]] = {}
        self.tools: list[BaseTool] = []
        self.failed: dict[str, str] = {}

    @property
    def names(self) -> list[str]:
        return list(self._config)

    async def connect(self, store: BaseStore | None = None) -> list[BaseTool]:
        """Connect to all configured MCP servers and fetch their tools.

        Config comes from the durable store when it has entries (CRUD API),
        otherwise from env/file (`MCP_SERVERS_JSON` / `mcp_servers.json`).
        """
        self._config = (
            await load_tool_server_configs(store)
            if store is not None
            else settings.load_mcp_servers()
        )
        if not self._config:
            logger.info("No MCP servers configured")
            return []

        client = MultiServerMCPClient(self._config)
        self.tools = await client.get_tools()
        logger.info("Connected MCP servers: %s (%d tools total)", self.names, len(self.tools))
        return self.tools

    async def close(self) -> None:
        self.tools = []


mcp_servers = MCPServers()
