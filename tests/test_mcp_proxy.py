"""Offline tests for the MCP apps tools proxy (POST /mcp/tools/call).

Spawns two real FastMCP stdio servers (same wire protocol as gofastmcp)
with a stateful backend tool, then drives the endpoint over ASGI:

  - CallToolResult passthrough (text content + structuredContent)
  - stateful tool across calls (cached, multiplexed sessions)
  - upstream isError -> HTTP 200 with isError: true
  - fan-out across servers without server_hint (first hit wins)
  - no match -> 404, unknown hint -> 404, auth required, name validation
  - JSON-RPC error paths (new-FastMCP/gofastmcp style) via stub sessions
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent
from pydantic import Field

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.mcp import MCPServers, McpToolNotFoundError, McpTransportError, mcp_servers

pytestmark = [
    pytest.mark.filterwarnings(r"ignore:The v3 streaming protocol on Pregel is experimental."),
    # Module-scoped fixture (app + stdio MCP servers) shares one event loop
    # with every test in this module; otherwise anyio cancel scopes from the
    # fixture's loop break when sessions are used from per-test loops.
    pytest.mark.asyncio(loop_scope="module"),
]


class Scripted(BaseChatModel):
    """Returns a scripted sequence of AIMessages, clamping at the last."""

    responses: list[AIMessage] = Field(default_factory=list)
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = min(self._idx, len(self.responses) - 1)
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])


def scripted_model() -> Scripted:
    return Scripted(responses=[AIMessage(content="Final answer from the agent.")])


# 12-hex-char hash prefix, matching FastMCP's hashed app-tool wire format
# (`<hash>_<name>`); FastMCP resolves the hash server-side via get_tool_by_hash.
HASHED_SAVE_CONTACT = "a1b2c3d4e5f6_save_contact"


def test_tool_not_found_detection():
    """Non-FastMCP servers (context7) answer "Tool <name> not found" — must
    count as tool-not-found so fan-out skips them instead of 502ing."""
    from app.services.mcp import _is_tool_not_found

    assert _is_tool_not_found("Tool f15d2d6be326_submit_answer not found")
    assert _is_tool_not_found("Tool submit_answer not found")
    # Existing server dialects keep matching.
    assert _is_tool_not_found("tool not found")
    assert _is_tool_not_found("Not found: 'save_contact'")
    assert _is_tool_not_found("Unknown tool: save_contact")
    # Legit execution errors must NOT be treated as tool-not-found.
    assert not _is_tool_not_found("file /tmp/x not found")
    assert not _is_tool_not_found("invalid params")


# A real FastMCP server, spawned as a stdio subprocess. `role` selects which
# tools are registered; the beta server's tool is stateful (file-backed) so
# tests can observe that sessions are cached and reused across calls.
SERVER_SCRIPT = """\
import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

role = sys.argv[1]
state_dir = Path(sys.argv[2])

if role == "alpha":
    # A context7-style server: NOT FastMCP. Unknown tools are rejected with
    # a JSON-RPC -32000 error "Tool <name> not found" (no get_tool_by_hash
    # extension), which the backend must classify as tool-not-found so
    # fan-out skips this server instead of failing with a 502.
    import anyio

    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData, Tool

    server = Server("alpha")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="ping",
                description="Ping.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="fail_tool",
                description="Explicit upstream error (isError result).",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent] | CallToolResult:
        if name == "ping":
            return [TextContent(type="text", text="pong")]
        if name == "fail_tool":
            return CallToolResult(
                content=[TextContent(type="text", text="boom failed")], isError=True
            )
        raise McpError(ErrorData(code=-32000, message=f"Tool {name} not found"))

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
    sys.exit(0)

mcp = FastMCP("beta")


def load_state():
    f = state_dir / "contacts.json"
    return json.loads(f.read_text()) if f.exists() else {"contacts": []}


if role == "beta":

    @mcp.tool(name="__HASHED_SAVE_CONTACT__")
    def save_contact(email: str, name: str = "") -> CallToolResult:
        \"\"\"Save a contact; state persists across calls.\"\"\"
        state = load_state()
        state["contacts"].append({"email": email, "name": name})
        (state_dir / "contacts.json").write_text(json.dumps(state))
        return CallToolResult(
            content=[TextContent(type="text", text=f"Saved contact {email}")],
            structuredContent={"saved": len(state["contacts"]), "email": email},
        )

    @mcp.tool()
    def boom() -> str:
        \"\"\"Raise -> FastMCP wraps it into an isError result.\"\"\"
        raise RuntimeError("kaboom from beta")


if __name__ == "__main__":
    mcp.run(transport="stdio")
""".replace("__HASHED_SAVE_CONTACT__", HASHED_SAVE_CONTACT)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def proxy_env(tmp_path_factory):
    """App + two stdio MCP servers registered through the CRUD API (store-first)."""
    script_dir = tmp_path_factory.mktemp("mcp-servers")
    script = script_dir / "test_server.py"
    script.write_text(SERVER_SCRIPT)
    state_dir = script_dir / "state"
    state_dir.mkdir()

    config.settings.database_uri = None
    await database.persistence.start()
    app = create_app(
        agent=build_agent(
            checkpointer=database.persistence.checkpointer,
            store=database.persistence.store,
            model=scripted_model(),
            system_prompt="test",
        )
    )
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()

    await database.persistence.users.create_user(
        username="tester", hashed_password="x", role="admin"
    )
    token = create_access_token(data={"sub": "tester"})
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )
    for name, role in (("alpha", "alpha"), ("beta", "beta")):
        r = await http.post(
            "/agent/tools",
            json={
                "name": name,
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(script), role, str(state_dir)],
            },
        )
        assert r.status_code == 201, r.text
    # Same path as POST /agent/tools/reconnect (store configs -> live servers).
    await mcp_servers.connect(store=database.persistence.store)
    try:
        yield http, state_dir
    finally:
        await http.aclose()
        await lifespan_cm.__aexit__(None, None, None)
        await database.persistence.stop()


async def call(
    http: httpx.AsyncClient,
    name: str,
    arguments: dict | None = None,
    server_hint: str | None = None,
    headers: dict | None = None,
):
    return await http.post(
        "/mcp/tools/call",
        json={
            "name": name,
            "arguments": arguments or {},
            **({"server_hint": server_hint} if server_hint else {}),
        },
        headers=headers,
    )


# ---------------------------------------------------------------------------
# connect() failure isolation
# ---------------------------------------------------------------------------


async def test_connect_isolates_unreachable_server(tmp_path, monkeypatch):
    """One unreachable server must not prevent the healthy ones from loading.

    Regression test: `fastmcp` pointed at 127.0.0.1 from inside the app
    container used to abort connect() entirely (httpx.ConnectError), so
    context7/grep tools never loaded and reconnect returned a raw 500.
    """
    script_dir = tmp_path / "servers"
    script_dir.mkdir()
    script = script_dir / "test_server.py"
    script.write_text(SERVER_SCRIPT)
    state_dir = script_dir / "state"
    state_dir.mkdir()

    monkeypatch.setattr(
        config.settings,
        "load_mcp_servers",
        lambda: {
            "alpha": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(script), "alpha", str(state_dir)],
            },
            "dead": {
                "url": "http://127.0.0.1:1/mcp",
                "transport": "streamable_http",
            },
        },
    )
    servers = MCPServers()
    tools = await servers.connect(store=None)

    assert "alpha" in servers.tools_by_server
    assert tools, "healthy server's tools must still load"
    assert "dead" in servers.failed
    assert servers.tools_by_server.get("dead") is None
    assert any(t.name == "ping" for t in tools)


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


async def test_tools_call_passthrough(proxy_env):
    http, _state_dir = proxy_env
    r = await call(http, HASHED_SAVE_CONTACT, {"email": "a@b.c", "name": "Ana"}, "beta")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == [{"type": "text", "text": "Saved contact a@b.c"}]
    assert body["structuredContent"] == {"saved": 1, "email": "a@b.c"}
    assert body["isError"] is False


async def test_plain_tool_name_passthrough(proxy_env):
    """Un-hashed MCP tool names pass through too."""
    http, _ = proxy_env
    r = await call(http, "ping", {}, "alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert any(block.get("text") == "pong" for block in body["content"])
    assert body["isError"] is False


async def test_stateful_tool_reuses_session(proxy_env):
    """Calls hit the same cached server session (state persists, no reconnect)."""
    http, state_dir = proxy_env
    r = await call(http, HASHED_SAVE_CONTACT, {"email": "b@c.d"}, "beta")
    assert r.json()["structuredContent"]["saved"] == 2
    r = await call(http, HASHED_SAVE_CONTACT, {"email": "c@d.e"}, "beta")
    assert r.json()["structuredContent"]["saved"] == 3
    contacts = json.loads((state_dir / "contacts.json").read_text())["contacts"]
    assert [c["email"] for c in contacts] == ["a@b.c", "b@c.d", "c@d.e"]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


async def test_is_error_result_passthrough(proxy_env):
    """Upstream isError: true -> HTTP 200 with isError: true (renderer handles it)."""
    http, _ = proxy_env
    r = await call(http, "fail_tool", {}, "alpha")
    assert r.status_code == 200, r.text
    assert r.json()["isError"] is True
    assert r.json()["content"] == [{"type": "text", "text": "boom failed"}]


async def test_is_error_raised_tool_wrapped(proxy_env):
    """A raising tool is wrapped by the server into an isError result (200)."""
    http, _ = proxy_env
    r = await call(http, "boom", {}, "beta")
    assert r.status_code == 200, r.text
    assert r.json()["isError"] is True
    assert "kaboom from beta" in r.json()["content"][0]["text"]


async def test_fanout_first_hit_wins(proxy_env):
    """No server_hint: alpha (context7-style) reports tool-not-found, beta
    executes (first hit wins). Regression for the real incident where a
    hashed app-tool name fanned out to context7 and the raw -32000 error
    leaked as a 502 instead of skipping to the FastMCP server behind it."""
    http, _ = proxy_env
    r = await call(http, HASHED_SAVE_CONTACT, {"email": "d@e.f"})
    assert r.status_code == 200, r.text
    assert r.json()["structuredContent"]["saved"] == 4
    assert r.json()["isError"] is False


async def test_fanout_is_error_first_hit_wins(proxy_env):
    """A real upstream error is authoritative: no fan-out past the owning server."""
    http, _ = proxy_env
    r = await call(http, "fail_tool")
    assert r.status_code == 200, r.text
    assert r.json()["isError"] is True
    assert r.json()["content"] == [{"type": "text", "text": "boom failed"}]


async def test_no_match_404(proxy_env):
    """No server knows the name -> 404 (with the servers tried)."""
    http, _ = proxy_env
    r = await call(http, "ffffffffffff_ghost_tool")
    assert r.status_code == 404, r.text
    assert "ffffffffffff_ghost_tool" in r.json()["detail"]
    assert "alpha" in r.json()["detail"] and "beta" in r.json()["detail"]


async def test_hint_wrong_server_404(proxy_env):
    """server_hint is authoritative: a miss on the hinted server is a 404 (no fallback)."""
    http, _ = proxy_env
    r = await call(http, HASHED_SAVE_CONTACT, {"email": "x@y.z"}, "alpha")
    assert r.status_code == 404, r.text


async def test_unknown_hint_404(proxy_env):
    http, _ = proxy_env
    r = await call(http, "ping", {}, "ghost")
    assert r.status_code == 404, r.text
    assert "ghost" in r.json()["detail"]


# ---------------------------------------------------------------------------
# security & validation
# ---------------------------------------------------------------------------


async def test_auth_required(proxy_env):
    http, _ = proxy_env
    r = await call(http, "ping", {}, "alpha", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401, r.text


async def test_invalid_tool_name_422(proxy_env):
    http, _ = proxy_env
    for bad in ("Bad Name!", "hash tool", "a" * 129, ""):
        r = await call(http, bad, {}, "alpha")
        assert r.status_code == 422, (bad, r.status_code, r.text)


# ---------------------------------------------------------------------------
# JSON-RPC error paths (gofastmcp / current FastMCP style) — stub sessions
# ---------------------------------------------------------------------------


class StubSession:
    def __init__(self, outcome):
        self.outcome = outcome  # CallToolResult or Exception to raise
        self.calls = 0

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _servers_with_sessions(outcomes: dict[str, StubSession]) -> MCPServers:
    servers = MCPServers()
    servers._config = {
        name: {"transport": "stdio", "command": "x", "args": []} for name in outcomes
    }
    servers._sessions = dict(outcomes)
    return servers


async def test_mcp_error_not_found_fans_out():
    """JSON-RPC 'not found' errors (gofastmcp / current FastMCP) drive the fan-out."""
    not_found = McpError(ErrorData(code=-32001, message="Not found: 'a1b2c3d4e5f6_save_contact'"))
    hit = CallToolResult(content=[TextContent(type="text", text="Saved")])
    alpha, beta = StubSession(not_found), StubSession(hit)
    servers = _servers_with_sessions({"alpha": alpha, "beta": beta})
    out = await servers.call_tool(HASHED_SAVE_CONTACT, {"email": "a@b.c"})
    assert alpha.calls == 1 and beta.calls == 1
    assert out.content == [{"type": "text", "text": "Saved"}]
    assert out.isError is False


async def test_mcp_error_all_not_found_404():
    not_found = McpError(ErrorData(code=-32001, message="Not found: 'x'"))
    servers = _servers_with_sessions(
        {"alpha": StubSession(not_found), "beta": StubSession(not_found)}
    )
    with pytest.raises(McpToolNotFoundError):
        await servers.call_tool("x")


async def test_mcp_error_transport_failure_502():
    """Non-not-found protocol errors are transport failures (502), not fan-out triggers."""
    internal = McpError(ErrorData(code=-32603, message="Internal error"))
    alpha, beta = StubSession(internal), StubSession(CallToolResult(content=[]))
    servers = _servers_with_sessions({"alpha": alpha, "beta": beta})
    with pytest.raises(McpTransportError):
        await servers.call_tool("x")
    assert beta.calls == 0  # aborted on the first real failure
    assert "alpha" not in servers._sessions  # dead session evicted for reconnect


async def test_transport_exception_evicts_session():
    """A generic transport exception drops the cached session so the next call reconnects."""
    alpha = StubSession(RuntimeError("connection reset"))
    servers = _servers_with_sessions({"alpha": alpha})
    with pytest.raises(McpTransportError):
        await servers.call_tool("x")
    assert "alpha" not in servers._sessions
