"""Query rewriting (R4): turn vague user queries into better retrieval queries.

An optional LLM call before retrieval, per the RAG research: rewrite helps
when the user query is vague/elliptical and the corpus speaks a different
vocabulary. Guardrails:

- skipped for trivial queries (too short or a single token — keyword search
  handles those better; rewrite adds latency without signal)
- content-addressed cache (temperature 0 → deterministic; capped size)
- any failure degrades to the original query (logged)
- opt-in via KB_QUERY_REWRITE; the agent's own search loop already adapts
  queries between tool calls, so rewriting mainly helps the REST endpoints
"""

from __future__ import annotations

import logging
from typing import Protocol

from ...core.config import settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
Rewrite this search query into a clearer, more specific query for searching a
knowledge base of technical documents (runbooks, code, ops docs). Keep all
exact terms, error codes, product names and identifiers unchanged. Add
missing context only when the query is vague or elliptical. Reply with ONLY
the rewritten query, no explanation.

Original: {query}
Rewritten:"""


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        """A better retrieval query; may return the original unchanged."""
        ...


class IdentityQueryRewriter:
    """No rewriting: the user query is used as-is."""

    def rewrite(self, query: str) -> str:
        return query


class LLMQueryRewriter:
    """LLM rewrite with trivial-query gate + per-query cache + safe fallback."""

    def __init__(self, model: str, min_length: int = 8, cache_size: int = 1024) -> None:
        self._model = model
        self._min_length = min_length
        self._cache_size = cache_size
        self._llm = None
        self._cache: dict[str, str] = {}

    def _ensure_llm(self):
        if self._llm is None:
            from langchain.chat_models import init_chat_model

            self._llm = init_chat_model(self._model, temperature=0)
        return self._llm

    def rewrite(self, query: str) -> str:
        query = query.strip()
        if len(query) < self._min_length or " " not in query:
            return query  # trivial: keyword/vector search handles it directly
        if query in self._cache:
            return self._cache[query]
        try:
            llm = self._ensure_llm()
            rewritten = llm.invoke(REWRITE_PROMPT.format(query=query)).content.strip()
        except Exception:
            logger.exception("query rewrite failed; using the original query")
            return query
        if not rewritten:
            return query
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[query] = rewritten
        return rewritten


def build_rewriter(*, force: bool = False) -> QueryRewriter:
    """Configured rewriter: LLM when KB_QUERY_REWRITE is on, else identity.

    `force=True` builds the LLM rewriter regardless (eval scripts comparing
    rewrite vs plain retrieval on a golden set).
    """
    if force or settings.kb_query_rewrite:
        model = settings.kb_rewrite_model or settings.model
        if model:
            return LLMQueryRewriter(
                model=model,
                min_length=settings.kb_rewrite_min_length,
            )
    # No configured LLM (KB_QUERY_REWRITE on but neither DEEPAGENTS_MODEL
    # nor KB_REWRITE_MODEL set) -> plain retrieval instead of crashing.
    return IdentityQueryRewriter()


# ---------------------------------------------------------------------------
# singleton wiring (tests swap in a deterministic rewriter)
# ---------------------------------------------------------------------------

_configured: QueryRewriter | None = None
_built = False


def get_rewriter() -> QueryRewriter:
    """The process-wide rewriter, built once from settings."""
    global _configured, _built
    if not _built:
        _configured = build_rewriter()
        _built = True
    return _configured


def set_rewriter(rewriter: QueryRewriter | None) -> None:
    """Replace the singleton (tests inject a fake rewriter)."""
    global _configured, _built
    _configured = rewriter
    _built = rewriter is not None


def reset_rewriter() -> None:
    """Drop the singleton so the next get_rewriter() rebuilds from settings."""
    global _configured, _built
    _configured = None
    _built = False
