"""Reranking stage (R3): retrieve broad, rerank fine.

The winning funnel from the RAG research: hybrid search pulls a wide
candidate pool (recall), a cross-encoder reranks query-chunk pairs
(precision), the top-k survives. Latency of a small CPU cross-encoder is
~30ms for 20 candidates vs ~10s LLM generation — negligible.

Implementations:
- `IdentityReranker` — no-op (default; KB_RERANK_MODEL unset)
- `FlashRankReranker` — local CPU cross-encoder (onnxruntime, model ~4MB,
  downloaded lazily from HuggingFace on first use; failures degrade to
  identity with a logged warning)
- a fake/deterministic variant is provided for offline tests via
  `set_reranker()`

The reranker scores the **same text that was embedded** (raw chunk content
today; if contextual retrieval is added later, both must be updated
together — the "rerank what you embed" law).
"""

from __future__ import annotations

import logging
from typing import Protocol

from ...core.config import settings
from ...schema.kb_schema import SearchHit

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Return the best `top_k` hits for the query, best first."""
        ...


class IdentityReranker:
    """No reranking: the vector store's own ranking is kept."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return hits[:top_k]


class FlashRankReranker:
    """Local CPU cross-encoder via flashrank (ONNX, no GPU needed)."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._ranker = None

    def _ensure(self):
        if self._ranker is None:
            from flashrank import Ranker

            self._ranker = Ranker(model_name=self._model, cache_dir="/tmp")
            logger.info("FlashRank reranker ready (model=%s)", self._model)
        return self._ranker

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        try:
            from flashrank import RerankRequest

            ranker = self._ensure()
            passages = [{"id": index, "text": hit.content} for index, hit in enumerate(hits)]
            results = ranker.rerank(RerankRequest(query=query, passages=passages))
            return [hits[int(r["id"])] for r in results[:top_k]]
        except Exception:
            logger.exception("reranking failed; falling back to the store ranking")
            return hits[:top_k]


def build_reranker() -> Reranker:
    """Configured reranker: FlashRank when KB_RERANK_MODEL is set, else identity."""
    if settings.kb_rerank_model:
        return FlashRankReranker(model=settings.kb_rerank_model)
    return IdentityReranker()


def search_with_rerank(
    store,
    reranker: Reranker,
    query: str,
    *,
    owner: str,
    kb_id: str | None = None,
    limit: int = 5,
    alpha: float = 0.5,
    candidates: int | None = None,
    rewriter=None,
) -> list[SearchHit]:
    """Rewrite -> hybrid retrieve (broad) -> rerank (fine) in one call.

    With identity reranker/rewriter this is exactly a plain `store.search` at
    the final limit, so behavior is unchanged unless the stages are enabled.
    """
    from .rewrite import IdentityQueryRewriter

    if rewriter is not None and not isinstance(rewriter, IdentityQueryRewriter):
        query = rewriter.rewrite(query)
    if reranker is None or isinstance(reranker, IdentityReranker):
        return store.search(query, owner=owner, kb_id=kb_id, limit=limit, alpha=alpha)
    pool = max(candidates or settings.kb_rerank_candidates, limit)
    broad = store.search(query, owner=owner, kb_id=kb_id, limit=pool, alpha=alpha)
    if not broad:
        return []
    return reranker.rerank(query, broad, limit)


# ---------------------------------------------------------------------------
# singleton wiring (tests swap in a deterministic reranker)
# ---------------------------------------------------------------------------

_configured: Reranker | None = None
_built = False


def get_reranker() -> Reranker:
    """The process-wide reranker, built once from settings."""
    global _configured, _built
    if not _built:
        _configured = build_reranker()
        _built = True
    return _configured


def set_reranker(reranker: Reranker | None) -> None:
    """Replace the singleton (tests inject a fake reranker)."""
    global _configured, _built
    _configured = reranker
    _built = reranker is not None


def reset_reranker() -> None:
    """Drop the singleton so the next get_reranker() rebuilds from settings."""
    global _configured, _built
    _configured = None
    _built = False
