"""SearXNG web search: a toggleable `web_search` tool for the agent.

The tool queries a self-hosted SearXNG metasearch instance's JSON API
(`/search?format=json`). Three toggle levels:

1. **Install**: `SEARXNG_URL` set in `.env` — otherwise the tool is not
   registered on the agent at all (zero overhead, invisible to the model).
2. **Global**: `SEARXNG_ENABLED=true|false` (default `true` when URL is set) —
   the tool exists but returns a "disabled" message when off.
3. **Per request**: `enable_search` (`/chat`) / `enableSearch` (`/api/chat`)
   field in the request body, applied via a contextvar read at tool call time —
   lets the frontend flip search per message.

JSON format must be enabled on the instance (`search.formats: [html, json]`,
see `searxng/settings.yml` mounted into the docker-compose service).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ..core.config import settings

logger = logging.getLogger(__name__)

TimeRange = Literal["day", "month", "year"]

DISABLED_MESSAGE = "Web search is disabled for this request."

# Per-request override; None = fall back to the global config toggle.
_search_override: ContextVar[bool | None] = ContextVar("searxng_search_override", default=None)


def set_search_enabled(value: bool | None) -> None:
    """Override the SearXNG toggle for the current request (contextvar)."""
    _search_override.set(value)


def search_allowed() -> bool:
    """Effective toggle: per-request override if set, else the global config."""
    override = _search_override.get()
    return settings.searxng_enabled if override is None else override


class SearxngClient:
    """Async client for a SearXNG instance's JSON API."""

    def __init__(
        self,
        base_url: str,
        *,
        max_results: int = 5,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_results = max_results
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def search(
        self,
        query: str,
        *,
        categories: str = "general",
        time_range: TimeRange | None = None,
        language: str | None = None,
    ) -> str:
        """Run a search and return results formatted for the LLM to read."""
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": categories,
            "safesearch": 1,
        }
        if time_range:
            params["time_range"] = time_range
        if language:
            params["language"] = language

        try:
            resp = await self._client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("SearXNG request failed: %s", exc)
            return f"Web search failed (SearXNG unreachable): {exc}"
        except ValueError as exc:
            logger.warning("SearXNG returned invalid JSON: %s", exc)
            return f"Web search failed (invalid response from SearXNG): {exc}"

        results = data.get("results") or []
        if not results:
            return "No results found."

        lines: list[str] = []
        # Numbered results: the model cites sources as [n] matching these
        # indices (see the citation instruction in DEFAULT_SYSTEM_PROMPT),
        # and the frontend parses the URLs from the markdown lines.
        for i, r in enumerate(results[: self.max_results], start=1):
            title = r.get("title") or "(untitled)"
            url = r.get("url") or ""
            content = (r.get("content") or "").strip()
            engines = r.get("engines") or ([r.get("engine")] if r.get("engine") else [])
            meta = ", ".join(x for x in (r.get("publishedDate") or "", ", ".join(engines)) if x)
            line = f"{i}. [{title}]({url})" + (f" ({meta})" if meta else "")
            if content:
                line += f"\n  {content}"
            lines.append(line)

        shown = len(results[: self.max_results])
        total = data.get("number_of_results") or "?"
        return f"{shown} result(s) (of {total}):\n" + "\n".join(lines)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class WebSearchInput(BaseModel):
    """Arguments for the `web_search` tool."""

    query: str = Field(description="Search query, e.g. 'latest FastAPI release notes'")
    categories: str = Field(
        default="general",
        description="SearXNG category (comma-separated): general, science, it, news, images, videos, ...",
    )
    time_range: TimeRange | None = Field(
        default=None, description="Only results from the past day, month, or year"
    )
    language: str | None = Field(
        default=None, description="ISO 639-1 language code, e.g. 'en' (default: instance default)"
    )


def build_search_tool(*, client: httpx.AsyncClient | None = None) -> BaseTool | None:
    """Build the `web_search` tool, or None when SEARXNG_URL is not configured.

    Args:
        client: optional injected httpx client (tests); a default one is created
            and owned by the tool otherwise.
    """
    if not settings.searxng_url:
        return None
    searxng = SearxngClient(
        settings.searxng_url,
        max_results=settings.searxng_max_results,
        timeout=settings.searxng_timeout,
        client=client,
    )

    async def _web_search(
        query: str,
        categories: str = "general",
        time_range: TimeRange | None = None,
        language: str | None = None,
    ) -> str:
        if not search_allowed():
            return DISABLED_MESSAGE
        return await searxng.search(
            query, categories=categories, time_range=time_range, language=language
        )

    return StructuredTool.from_function(
        coroutine=_web_search,
        name="web_search",
        description=(
            "Search the web for current information via a self-hosted SearXNG "
            "metasearch instance. Use for up-to-date facts, documentation, news, "
            "prices, and anything beyond your training data."
        ),
        args_schema=WebSearchInput,
    )
