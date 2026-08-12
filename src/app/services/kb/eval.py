"""Retrieval evaluation: golden set + IR metrics (Recall@k, MRR, nDCG).

Used by `scripts/kb_eval.py` (CLI sweep) and offline tests. A golden query
maps a search query to the document paths that are relevant (ground truth,
curated per KB by a human).

Metrics (binary relevance at document-path level, per query):
- Recall@k: fraction of relevant paths present in the top-k hits
- MRR: reciprocal rank of the first relevant hit (0 when none found)
- nDCG@k: DCG of the hit ranking normalized by the ideal ranking

The cardinal rule of the RAG roadmap: every technique change must be
measured against this baseline before being adopted.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field

from .vectorstore import KbVectorStore

GOLDEN_SCHEMA = {
    "description": "curated golden set for scripts/kb_eval.py",
    "queries": [
        {
            "query": "a natural-language question a user might ask",
            "relevant": ["relative/doc/path.md"],
        }
    ],
}


@dataclass
class GoldenQuery:
    query: str
    relevant: list[str]


@dataclass
class QueryEval:
    query: str
    relevant: list[str]
    top_hits: list[str] = field(default_factory=list)
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0


def load_golden(path: str) -> list[GoldenQuery]:
    """Load and validate a golden set JSON file (see GOLDEN_SCHEMA)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
        raise ValueError("Golden set must be an object with a 'queries' list")
    queries: list[GoldenQuery] = []
    for item in raw["queries"]:
        query = item.get("query") if isinstance(item, dict) else None
        relevant = item.get("relevant") if isinstance(item, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each golden query needs a non-empty 'query' string")
        if not isinstance(relevant, list) or not all(isinstance(p, str) for p in relevant):
            raise ValueError(f"Golden query {query!r} needs a 'relevant' list of paths")
        queries.append(GoldenQuery(query=query.strip(), relevant=relevant))
    if not queries:
        raise ValueError("Golden set has no queries")
    return queries


def _recall_at_k(hits: list, relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for hit in hits if hit.path in relevant)
    return found / len(relevant)


def _mrr(hits: list, relevant: set[str]) -> float:
    for index, hit in enumerate(hits):
        if hit.path in relevant:
            return 1.0 / (index + 1)
    return 0.0


def _ndcg_at_k(hits: list, relevant: set[str], k: int | None = None) -> float:
    """nDCG@k with binary relevance; ideal DCG is capped at min(k, #relevant)."""
    k = k if k is not None else len(hits)
    rel = [1 if hit.path in relevant else 0 for hit in hits[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    store: KbVectorStore,
    queries: list[GoldenQuery],
    *,
    owner: str,
    alpha: float,
    limit: int,
) -> dict:
    """Evaluate one configuration; returns aggregates + per-query details.

    `owner` must match the owner the corpus was upserted with (multi-tenant
    invariant: evaluation never bypasses the owner filter).
    """
    per_query: list[QueryEval] = []
    for golden in queries:
        relevant = set(golden.relevant)
        hits = store.search(golden.query, owner=owner, limit=limit, alpha=alpha)
        per_query.append(
            QueryEval(
                query=golden.query,
                relevant=list(golden.relevant),
                top_hits=[h.path for h in hits],
                recall_at_k=_recall_at_k(hits, relevant),
                mrr=_mrr(hits, relevant),
                ndcg_at_k=_ndcg_at_k(hits, relevant),
            )
        )
    return {
        "alpha": alpha,
        "limit": limit,
        "queries": len(per_query),
        "recall_at_k": statistics.fmean(q.recall_at_k for q in per_query),
        "mrr": statistics.fmean(q.mrr for q in per_query),
        "ndcg_at_k": statistics.fmean(q.ndcg_at_k for q in per_query),
        "per_query": per_query,
    }
