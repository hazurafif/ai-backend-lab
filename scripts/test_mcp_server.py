"""Tiny MCP server for testing backend MCP integration.

Same protocol as a gofastmcp (Go) server: exposes tools over streamable HTTP
at http://127.0.0.1:8090/mcp. Run: uv run python scripts/test_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-demo")


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city. Returns structured data."""
    return {
        "city": city,
        "temperature_c": 24,
        "condition": "sunny",
        "humidity": 40,
        "units": "metric",
    }


@mcp.tool()
def list_cities() -> list[str]:
    """List cities available in the demo."""
    return ["tokyo", "paris", "berlin", "jakarta"]


if __name__ == "__main__":
    # streamable HTTP at http://127.0.0.1:8090/mcp (same as gofastmcp servers)
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = 8090
    mcp.run(transport="streamable-http")
