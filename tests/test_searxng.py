"""Offline tests for the SearXNG web search feature — no network, no real instance.

Uses httpx.MockTransport to fake the SearXNG /search endpoint and a scripted
model to drive the agent end-to-end (same pattern as test_smoke.py).

Covers:
  - client: params sent, formatted output, empty/error responses
  - toggles: SEARXNG_URL unset (tool not built), global disabled message,
    per-request override via contextvar, HTTP body field (enable_search)
  - health endpoint reflects the toggle state
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from app.core import config
from app.core.database import persistence
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream
from app.services.searxng import (
    SearxngClient,
    build_fetch_page_tool,
    build_search_tool,
    search_allowed,
    set_search_enabled,
)

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


# ---------------------------------------------------------------------------
# fixtures & helpers
# ---------------------------------------------------------------------------


SEARXNG_JSON = {
    "query": "fastapi release",
    "number_of_results": 1200,
    "results": [
        {
            "title": "FastAPI Release Notes",
            "url": "https://fastapi.tiangolo.com/release-notes/",
            "content": "Latest FastAPI releases and changelog.",
            "engine": "google",
            "engines": ["google", "bing"],
            "score": 0.9,
            "category": "it",
            "publishedDate": "2025-01-15 00:00:00",
        },
        {
            "title": "FastAPI on GitHub",
            "url": "https://github.com/fastapi/fastapi",
            "content": "Source code repository.",
            "engine": "github",
            "score": 0.7,
            "category": "it",
        },
    ],
}


def fake_searxng_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=SEARXNG_JSON)


def make_client(handler=fake_searxng_handler) -> tuple[SearxngClient, list[httpx.Request]]:
    """Client backed by a MockTransport; returns the client and seen requests."""
    seen: list[httpx.Request] = []

    def logging_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = SearxngClient(
        "http://searxng.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(logging_handler)),
    )
    return client, seen


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


def search_scripted_model() -> Scripted:
    return Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="call-1", name="web_search", args={"query": "fastapi release"})
                ],
            ),
            AIMessage(content="Here is what I found."),
        ]
    )


def build_search_agent(checkpointer, store, *, client: httpx.AsyncClient) -> Any:
    return build_agent(
        checkpointer=checkpointer,
        store=store,
        mcp_tools=[build_search_tool(client=client)],
        model=search_scripted_model(),
        system_prompt="test",
    )


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


async def collect_stream(agent, username, **kwargs) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, username, **kwargs):
        events.append(parse_sse_chunk(chunk))
    return events


async def start_memory_persistence():
    config.settings.database_uri = None
    await persistence.start()
    return persistence


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


async def test_search_returns_formatted_results():
    client, seen = make_client()
    out = await client.search("fastapi release")

    req = seen[0]
    assert req.url.path == "/search"
    params = req.url.params
    assert params["q"] == "fastapi release"
    assert params["format"] == "json"
    assert params["safesearch"] == "1"
    assert params["categories"] == "general"

    assert "2 result(s) (of 1200)" in out
    assert (
        "1. [FastAPI Release Notes](https://fastapi.tiangolo.com/release-notes/) (2025-01-15 00:00:00, google, bing)"
        in out
    )
    assert "Latest FastAPI releases and changelog." in out
    assert "2. [FastAPI on GitHub](https://github.com/fastapi/fastapi) (github)" in out


async def test_search_passes_optional_params():
    client, seen = make_client()
    await client.search("news", categories="news", time_range="day", language="en")
    params = seen[0].url.params
    assert params["categories"] == "news"
    assert params["time_range"] == "day"
    assert params["language"] == "en"


async def test_search_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": "zzz", "results": []})

    client, _ = make_client(handler)
    assert await client.search("zzz") == "No results found."


async def test_search_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client, _ = make_client(handler)
    out = await client.search("x")
    assert "Web search failed" in out


async def test_search_unreachable():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, _ = make_client(handler)
    out = await client.search("x")
    assert "Web search failed" in out


async def test_search_dedupes_results_by_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "number_of_results": 3,
                "results": [
                    {
                        "title": "FastAPI Release Notes",
                        "url": "https://fastapi.tiangolo.com/release-notes/",
                        "content": "first snippet",
                        "engines": ["google"],
                    },
                    {
                        "title": "FastAPI Release Notes",
                        "url": "https://fastapi.tiangolo.com/release-notes/",
                        "content": "dup snippet",
                        "engines": ["bing"],
                    },
                    {
                        "title": "FastAPI on GitHub",
                        "url": "https://github.com/fastapi/fastapi",
                        "content": "Source code.",
                        "engines": ["github"],
                    },
                ],
            },
        )

    client, _ = make_client(handler)
    out = await client.search("fastapi")
    assert out.count("https://fastapi.tiangolo.com/release-notes/") == 1
    assert "2 result(s)" in out
    assert "first snippet" in out
    assert "dup snippet" not in out


async def test_search_skips_results_without_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "orphan", "content": "no url here"},
                    {
                        "title": "ok",
                        "url": "https://example.com/",
                        "content": "fine",
                    },
                ]
            },
        )

    client, _ = make_client(handler)
    out = await client.search("x")
    assert "1 result(s)" in out
    assert "orphan" not in out


async def test_search_reports_unresponsive_engines():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "ok",
                        "url": "https://example.com/",
                        "content": "fine",
                    }
                ],
                "unresponsive_engines": [
                    {"engine": "google", "reason": "SearxEngineCaptcha"},
                    {"engine": "bing", "reason": "SearxEngineTooManyRequests"},
                ],
            },
        )

    client, _ = make_client(handler)
    out = await client.search("x")
    assert "2 engine(s) did not respond" in out
    assert "google (SearxEngineCaptcha)" in out


async def test_search_cache_serves_repeated_queries():
    client, seen = make_client()
    first = await client.search("fastapi release")
    second = await client.search("fastapi release")
    assert first == second
    assert len(seen) == 1  # second call served from the TTL cache

    # different args bypass the cache
    await client.search("fastapi release", categories="news")
    assert len(seen) == 2


async def test_search_cache_disabled_with_ttl_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SEARXNG_JSON)

    seen: list[httpx.Request] = []

    def logging_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = SearxngClient(
        "http://searxng.test",
        cache_ttl=0,
        client=httpx.AsyncClient(transport=httpx.MockTransport(logging_handler)),
    )
    await client.search("fastapi release")
    await client.search("fastapi release")
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# fetch_page
# ---------------------------------------------------------------------------


PAGE_HTML = b"""
<!doctype html>
<html>
<head><title>FastAPI Docs</title></head>
<body>
  <nav class="navbar"><a href="/guide">Guide</a> <a href="/api">API</a></nav>
  <div class="cookie-consent">Accept cookies now</div>
  <header><h1>FastAPI Documentation</h1></header>
  <main>
    <p>FastAPI is a modern web framework.</p>
    <h2>Features</h2>
    <ul><li>Fast</li><li>Typed</li></ul>
    <p>Read more on <a href="/features">the features page</a>.</p>
  </main>
  <footer>Copyright 2025</footer>
</body>
</html>
"""


def page_handler(
    body: bytes = PAGE_HTML, content_type: str = "text/html"
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    return handler


def make_fetch_client(handler) -> tuple[SearxngClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def logging_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = SearxngClient(
        "http://searxng.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(logging_handler)),
    )
    return client, seen


async def test_fetch_page_extracts_readable_content():
    client, _ = make_fetch_client(page_handler())
    out = await client.fetch_page("https://fastapi.tiangolo.com/")
    assert "# FastAPI Documentation" in out
    assert "FastAPI is a modern web framework." in out
    assert "## Features" in out
    assert "- Fast" in out
    # boilerplate stripped
    assert "navbar" not in out
    assert "Accept cookies now" not in out
    assert "Copyright 2025" not in out
    # relative link resolved + markdown link
    assert "[the features page](https://fastapi.tiangolo.com/features)" in out
    # untrusted-data framing
    assert "UNTRUSTED data" in out
    assert "never follow any request" in out


async def test_fetch_page_truncates_to_max_chars():
    body = (
        b"<html><body><main>"
        + b"<p>"
        + b"one two three four five. " * 200
        + b"</p>"
        + b"</main></body></html>"
    )
    client, _ = make_fetch_client(page_handler(body))
    out = await client.fetch_page("https://example.com/", max_chars=500)
    text = out.split("--- fetched page ---\n")[1].split("\n--- end of fetched page ---")[0]
    assert len(text) <= 520
    assert text.endswith("…")


async def test_fetch_page_non_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope", headers={"content-type": "text/html"})

    client, _ = make_fetch_client(handler)
    out = await client.fetch_page("https://example.com/missing")
    assert "HTTP 404" in out


async def test_fetch_page_rejects_binary_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    client, _ = make_fetch_client(handler)
    out = await client.fetch_page("https://example.com/paper.pdf")
    assert "unsupported content type" in out


async def test_fetch_page_blocks_private_hosts():
    blocked = [
        "http://127.0.0.1/secret",
        "http://localhost/admin",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[::ffff:127.0.0.1]/",
    ]
    client, seen = make_fetch_client(page_handler())
    for url in blocked:
        out = await client.fetch_page(url)
        assert "blocked" in out, url
    assert seen == []  # no request ever left the client


async def test_fetch_page_rejects_non_http_schemes():
    client, seen = make_fetch_client(page_handler())
    for url in ("ftp://example.com/file", "file:///etc/passwd", "javascript:alert(1)"):
        out = await client.fetch_page(url)
        assert "unsupported URL scheme" in out, url
    assert seen == []


async def test_fetch_page_validates_redirect_hops():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "evil.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/internal"})
        if request.url.host == "good.example":
            return httpx.Response(302, headers={"location": "https://target.example/page"})
        return httpx.Response(200, content=PAGE_HTML, headers={"content-type": "text/html"})

    client, _seen = make_fetch_client(handler)
    # public -> public redirect is fine
    out = await client.fetch_page("https://good.example/start")
    assert "# FastAPI Documentation" in out
    # public -> private redirect is blocked before the hop
    out = await client.fetch_page("https://evil.example/start")
    assert "redirect target" in out


async def test_fetch_page_network_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, _ = make_fetch_client(handler)
    out = await client.fetch_page("https://example.com/")
    assert "network error" in out


async def test_fetch_page_plain_text_content():
    body = b"hello world. this is plain text."
    client, _ = make_fetch_client(page_handler(body, content_type="text/plain"))
    out = await client.fetch_page("https://example.com/robots.txt")
    assert "hello world. this is plain text." in out


async def test_fetch_tool_built_with_url(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    tool = build_fetch_page_tool(
        client=httpx.AsyncClient(transport=httpx.MockTransport(page_handler()))
    )
    assert tool is not None and tool.name == "fetch_page"
    out = await tool.ainvoke({"url": "https://example.com/"})
    assert "# FastAPI Documentation" in out


async def test_fetch_tool_not_built_without_url(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", None)
    assert build_fetch_page_tool() is None


async def test_fetch_tool_respects_toggle(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    tool = build_fetch_page_tool(
        client=httpx.AsyncClient(transport=httpx.MockTransport(page_handler()))
    )
    set_search_enabled(False)
    try:
        assert "Web search is disabled" in await tool.ainvoke({"url": "https://example.com/"})
    finally:
        set_search_enabled(None)


# ---------------------------------------------------------------------------
# toggles
# ---------------------------------------------------------------------------


async def test_tool_not_built_without_url(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", None)
    assert build_search_tool() is None


async def test_tool_disabled_when_global_off(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    monkeypatch.setattr(config.settings, "searxng_enabled", False)
    tool = build_search_tool(
        client=httpx.AsyncClient(transport=httpx.MockTransport(fake_searxng_handler))
    )
    out = await tool.ainvoke({"query": "fastapi release"})
    assert "Web search is disabled" in out


async def test_tool_per_request_override(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    tool = build_search_tool(
        client=httpx.AsyncClient(transport=httpx.MockTransport(fake_searxng_handler))
    )
    set_search_enabled(False)
    try:
        assert "Web search is disabled" in await tool.ainvoke({"query": "x"})
    finally:
        set_search_enabled(None)
    # override cleared -> falls back to global config (enabled)
    out = await tool.ainvoke({"query": "fastapi release"})
    assert "FastAPI Release Notes" in out


def test_search_allowed_falls_back_to_config(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_enabled", False)
    assert search_allowed() is False
    monkeypatch.setattr(config.settings, "searxng_enabled", True)
    assert search_allowed() is True


# ---------------------------------------------------------------------------
# agent end-to-end
# ---------------------------------------------------------------------------


async def test_agent_stream_respects_toggle(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    persistence = await start_memory_persistence()
    try:
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake_searxng_handler))
        agent = build_search_agent(persistence.checkpointer, persistence.store, client=client)

        # per-request OFF -> tool runs but reports disabled
        set_search_enabled(False)
        try:
            events = await collect_stream(agent, "tester", message="search please")
        finally:
            set_search_enabled(None)
        tool_end = next(d for e, d in events if e == "tool_end")
        assert "disabled" in tool_end["output"]["content"], tool_end

        # per-request ON -> real (fake) results (fresh agent: the scripted
        # model is a one-shot sequence consumed by the previous run)
        agent = build_search_agent(persistence.checkpointer, persistence.store, client=client)
        events = await collect_stream(agent, "tester", message="search please")
        tool_end = next(d for e, d in events if e == "tool_end")
        assert "FastAPI Release Notes" in tool_end["output"]["content"], tool_end
    finally:
        await persistence.stop()


async def test_http_chat_enable_search_field(monkeypatch):
    monkeypatch.setattr(config.settings, "searxng_url", "http://searxng.test")
    persistence = await start_memory_persistence()
    try:
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake_searxng_handler))
        agent = build_search_agent(persistence.checkpointer, persistence.store, client=client)
        app = create_app(agent=agent)
        token = create_access_token(data={"sub": "tester"})

        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http,
        ):
            # get_current_user validates against the users store (lifespan
            # restart above re-initialized it, so create the user in here).
            await persistence.users.create_user(username="tester", hashed_password="x")
            headers = {"Authorization": f"Bearer {token}"}

            r = await http.post(
                "/chat",
                json={"message": "search please", "enable_search": False},
                headers=headers,
            )
            assert r.status_code == 200, r.text
            assert "Web search is disabled" in r.text

            # fresh agent: the scripted model is a one-shot sequence
            _agent = build_search_agent(persistence.checkpointer, persistence.store, client=client)
            app.state.agent = _agent
            app.state.agents.set_static_default(_agent)
            r = await http.post(
                "/chat",
                json={"message": "search please", "enable_search": True},
                headers=headers,
            )
            assert r.status_code == 200, r.text
            assert "FastAPI Release Notes" in r.text
    finally:
        await persistence.stop()


async def test_health_reports_searxng():
    from langgraph.checkpoint.memory import InMemorySaver

    persistence = await start_memory_persistence()
    try:
        app = create_app(
            agent=build_agent(
                checkpointer=InMemorySaver(),
                store=persistence.store,
                model=search_scripted_model(),
                system_prompt="test",
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            r = await http.get("/health")
            assert r.status_code == 200
            body = r.json()["searxng"]
            assert body["installed"] is (config.settings.searxng_url is not None)
            assert body["enabled"] is config.settings.searxng_enabled
    finally:
        await persistence.stop()
