"""Offline tests for the MCP apps tools proxy (POST /mcp/tools/call).

Spawns two real FastMCP stdio servers (same wire protocol as gofastmcp)
with a stateful backend tool, then drives the endpoint over ASGI:

  - CallToolResult passthrough (text content + structuredContent)
  - stateful tool across calls (cached, multiplexed sessions)
  - upstream isError -> HTTP 200 with isError: true
  - fan-out across servers without server_hint (first hit wins)
  - no match -> 404, unknown hint -> 404, auth required, name validation
  - JSON-RPC error paths (new-FastMCP/gofastmcp style) via stub sessions

Plus the raw-SDK resources bridge: the beta server also exposes a static
resource and a resource template (same @mcp.resource wire format as
gofastmcp), covered by service-level tests (list/read, fan-out, error
mapping), bridge-tool invocations, and an end-to-end scripted-agent run.
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
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from mcp.shared.exceptions import McpError
from mcp.types import (
    CallToolResult,
    ErrorData,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
)
from pydantic import Field

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.mcp import (
    McpResourceNotFoundError,
    MCPServers,
    McpToolNotFoundError,
    McpTransportError,
    mcp_servers,
)

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
    tools: Sequence[dict | type] = ()
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

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseChatModel],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = tools
        return self


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

    @mcp.resource("data://greeting")
    def get_greeting() -> str:
        \"\"\"Provides a simple greeting message.\"\"\"
        return "Hello from FastMCP Resources!"

    @mcp.resource("weather://{city}/current")
    def get_weather(city: str) -> str:
        \"\"\"Provides weather information for a specific city.\"\"\"
        return json.dumps({"city": city.capitalize(), "temperature": 22, "condition": "Sunny"})


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
    await mcp_servers.connect("tester", store=database.persistence.store)
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
# resources: raw-SDK bridge (list + read, templates included)
# ---------------------------------------------------------------------------


async def test_connect_registers_resource_tools(proxy_env):
    """connect() exposes the resource bridge as agent tools, attributed to
    every connected server (so per-server tool selection keeps them)."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    names = [t.name for t in servers.tools]
    assert "mcp_list_resources" in names
    assert "mcp_read_resource" in names
    for server in ("alpha", "beta"):
        assert "mcp_list_resources" in servers.tools_by_server[server]
        assert "mcp_read_resource" in servers.tools_by_server[server]


async def test_list_resources_includes_templates(proxy_env):
    """Static resources AND RFC 6570 templates are discovered; a tool-only
    server (alpha: no resources feature, -32601) is skipped quietly, not
    recorded as a failure."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    items = await servers.list_resources()
    assert {i.server for i in items} == {"beta"}
    assert "alpha" not in servers.failed
    static = next(i for i in items if i.uri == "data://greeting")
    assert static.description == "Provides a simple greeting message."
    assert static.mime_type == "text/plain"
    tpl = next(i for i in items if i.uri_template == "weather://{city}/current")
    assert tpl.description == "Provides weather information for a specific city."


async def test_read_resource_static_and_template(proxy_env):
    """Reading works for static resources and template instantiations (the
    server resolves {param} matching; the client just sends the URI)."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    greeting = await servers.read_resource("data://greeting", server_hint="beta")
    assert greeting.content == [{"type": "text", "text": "Hello from FastMCP Resources!"}]
    weather = await servers.read_resource("weather://paris/current", server_hint="beta")
    assert "paris" in weather.content[0]["text"].lower()
    assert "temperature" in weather.content[0]["text"]


async def test_read_resource_fanout_skips_tool_only_server(proxy_env):
    """Without a hint, a server that never implements resources (-32601) is
    skipped and the next server answers (first hit wins)."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    out = await servers.read_resource("data://greeting")
    assert out.content == [{"type": "text", "text": "Hello from FastMCP Resources!"}]


async def test_read_resource_miss_raises(proxy_env):
    """server_hint is authoritative: a miss on the hinted server is an error;
    an unknown hint is rejected up front."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    with pytest.raises(McpResourceNotFoundError):
        await servers.read_resource("data://nope", server_hint="beta")
    with pytest.raises(McpToolNotFoundError):
        await servers.read_resource("data://greeting", server_hint="ghost")


async def test_resource_tools_invoke(proxy_env):
    """The bridge tools are callable as LangChain tools (the agent path):
    listing shows templates with placeholders, reading instantiates them,
    and misses surface as Error: strings (tools never raise)."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    by_name = {t.name: t for t in servers.tools}
    listing = await by_name["mcp_list_resources"].ainvoke({"server": "beta"})
    assert "data://greeting" in listing
    assert "weather://{city}/current" in listing
    read = await by_name["mcp_read_resource"].ainvoke(
        {"uri": "weather://paris/current", "server": "beta"}
    )
    assert "paris" in read.lower()
    missing = await by_name["mcp_read_resource"].ainvoke({"uri": "data://nope", "server": "beta"})
    assert missing.startswith("Error:")


async def test_agent_executes_resource_tool(proxy_env):
    """End-to-end: a scripted agent emits a tool call to mcp_read_resource and
    the bridge fetches a template-instantiated resource mid-conversation."""
    servers = await mcp_servers.get("tester", store=database.persistence.store)
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-res",
                        name="mcp_read_resource",
                        args={"uri": "weather://paris/current", "server": "beta"},
                    )
                ],
            ),
            AIMessage(content="It is sunny and 22C in Paris."),
        ]
    )
    agent = build_agent(
        checkpointer=database.persistence.checkpointer,
        store=database.persistence.store,
        mcp_tools=servers.tools,
        model=model,
        system_prompt="test",
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="What is the weather in Paris?")]},
        config={"configurable": {"thread_id": "resource-thread-1"}},
    )
    tool_msgs = [m for m in result["messages"] if getattr(m, "name", None) == "mcp_read_resource"]
    assert tool_msgs, [m.type for m in result["messages"]]
    assert "paris" in tool_msgs[0].content.lower()
    assert result["messages"][-1].content == "It is sunny and 22C in Paris."


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


class StubResourceSession:
    """Minimal stub with the resources surface used by the bridge."""

    def __init__(self, *, read=None, listed=None, templates=None):
        self._read = read  # ReadResourceResult | Exception
        self._listed = listed  # list[Resource] | Exception
        self._templates = templates  # list[ResourceTemplate] | Exception
        self.read_calls = 0

    async def list_resources(self):
        if isinstance(self._listed, Exception):
            raise self._listed
        return ListResourcesResult(resources=self._listed or [])

    async def list_resource_templates(self):
        if isinstance(self._templates, Exception):
            raise self._templates
        return ListResourceTemplatesResult(resource_templates=self._templates or [])

    async def read_resource(self, uri):
        self.read_calls += 1
        if isinstance(self._read, Exception):
            raise self._read
        return self._read


async def test_resource_not_found_fans_out():
    """-32002 resource-not-found drives the fan-out (first hit wins)."""
    not_found = McpError(ErrorData(code=-32002, message="Resource not found: data://nope"))
    hit = ReadResourceResult(contents=[TextResourceContents(uri="data://greeting", text="Hello")])
    alpha, beta = StubResourceSession(read=not_found), StubResourceSession(read=hit)
    servers = _servers_with_sessions({"alpha": alpha, "beta": beta})
    out = await servers.read_resource("data://greeting")
    assert alpha.read_calls == 1 and beta.read_calls == 1
    assert out.content == [{"type": "text", "text": "Hello"}]


async def test_resource_method_not_found_skips_server():
    """-32601 (server without the resources feature) is skipped in fan-out."""
    not_impl = McpError(ErrorData(code=-32601, message="Method not found"))
    hit = ReadResourceResult(contents=[TextResourceContents(uri="data://greeting", text="Hello")])
    alpha, beta = StubResourceSession(read=not_impl), StubResourceSession(read=hit)
    servers = _servers_with_sessions({"alpha": alpha, "beta": beta})
    out = await servers.read_resource("data://greeting")
    assert out.content == [{"type": "text", "text": "Hello"}]


async def test_resource_all_not_found_raises():
    not_found = McpError(ErrorData(code=-32002, message="Resource not found: x"))
    servers = _servers_with_sessions(
        {"alpha": StubResourceSession(read=not_found), "beta": StubResourceSession(read=not_found)}
    )
    with pytest.raises(McpResourceNotFoundError):
        await servers.read_resource("x")


async def test_resource_transport_failure_evicts_session():
    """Non-not-found protocol errors are transport failures (session evicted)."""
    internal = McpError(ErrorData(code=-32603, message="Internal error"))
    alpha = StubResourceSession(read=internal)
    servers = _servers_with_sessions({"alpha": alpha})
    with pytest.raises(McpTransportError):
        await servers.read_resource("x")
    assert "alpha" not in servers._sessions


async def test_resource_listing_isolates_dead_server():
    """A failing server is recorded in `failed` and skipped; others still list."""
    dead = RuntimeError("connection reset")
    alpha = StubResourceSession(listed=dead, templates=dead)
    beta = StubResourceSession()
    servers = _servers_with_sessions({"alpha": alpha, "beta": beta})
    items = await servers.list_resources()
    assert items == []
    assert "alpha" in servers.failed
    assert "alpha" not in servers._sessions  # dead session evicted
