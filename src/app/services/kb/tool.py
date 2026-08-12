"""Agent tool: `search_knowledge_base` — hybrid search over the user's KBs.

The tool resolves the current user from the langgraph runtime context
(`context={"user_id": ...}` passed by `services/chat.py`), so a user can only
ever see chunks of their own knowledge bases. Returns markdown-formatted hits
with source paths for citations.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, tool

from ...core.config import settings
from .rerank import get_reranker, search_with_rerank
from .rewrite import get_rewriter
from .vectorstore import KbVectorStore, get_vector_store

logger = logging.getLogger(__name__)

_MAX_CHUNK_CHARS = 600


def _current_user() -> str | None:
    """Resolve the user_id from the langgraph runtime context, if any."""
    try:
        from langgraph.runtime import get_runtime

        runtime = get_runtime()
    except Exception:
        return None
    if runtime is None:
        return None
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, dict):
        return ctx.get("user_id")
    return getattr(ctx, "user_id", None)


def build_kb_search_tool(vector_store: KbVectorStore | None = None) -> BaseTool | None:
    """The agent's KB search tool, or None when no vector store is configured.

    `vector_store` is for tests (inject the in-memory store); when omitted the
    configured singleton is resolved lazily at call time.
    """
    if vector_store is None and get_vector_store() is None:
        return None

    @tool
    def search_knowledge_base(query: str, kb_id: str | None = None, top_k: int = 5) -> str:
        """Search the user's knowledge bases for relevant passages.

        Use this when the user asks about content they uploaded to a knowledge
        base, or when answering from uploaded files (docs, PDFs, notes, code).
        Returns matching passages with their source file paths.

        Args:
            query: the question or keywords to search for.
            kb_id: restrict the search to one knowledge base (optional).
            top_k: how many passages to return (default 5).
        """
        store = vector_store or get_vector_store()
        if store is None:
            return "Knowledge base search is unavailable (vector store not configured)."
        user = _current_user()
        if not user:
            return "Knowledge base search is unavailable in this session."
        try:
            hits = search_with_rerank(
                store,
                get_reranker(),
                query,
                owner=user,
                kb_id=kb_id,
                limit=max(1, min(top_k, 20)),
                alpha=settings.kb_hybrid_alpha,
                rewriter=get_rewriter(),
            )
        except Exception as exc:
            logger.exception("knowledge base search failed")
            return f"Knowledge base search failed: {exc}"
        if not hits:
            return "No matching passages found in the knowledge base."
        # Dedup (same chunk reachable via different routes) and lost-in-the-
        # middle ordering: LLMs attend most to the start/end of context, so the
        # strongest evidence goes mid-list (research: rag-techniques-research.md).
        seen: set[tuple[str, int]] = set()
        unique: list = []
        for hit in hits:
            key = (hit.doc_id, hit.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            unique.append(hit)
        if len(unique) > 2:
            unique.insert(len(unique) // 2, unique.pop(0))
        lines = [f"Found {len(unique)} passage(s) from your knowledge base:"]
        for hit in unique:
            snippet = hit.content.strip().replace("\n", " ")[:_MAX_CHUNK_CHARS]
            lines.append(f"- `{hit.path}` (score {hit.score:.3f}): {snippet}")
        return "\n".join(lines)

    return search_knowledge_base


__all__ = ["build_kb_search_tool"]
