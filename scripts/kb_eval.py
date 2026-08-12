"""KB retrieval evaluation: golden set + IR metrics sweep (Recall@k, MRR, nDCG).

Builds a KB's corpus into a vector store (in-memory by default; --live uses
the configured Weaviate store) with the configured embeddings, then measures
retrieval quality across alpha values so tuning decisions are data-driven.

Usage:
    uv run python scripts/kb_eval.py --kb <id-or-name> --golden data/golden_set.json
    uv run python scripts/kb_eval.py --kb runbook --golden g.json --alphas 0,0.5,1.0
    uv run python scripts/kb_eval.py --kb runbook --golden g.json --live --verbose

Needs DATABASE_URI (Postgres) and OPENAI_API_KEY for meaningful numbers;
without them it still runs (in-memory + local embeddings) as a smoke test.

Golden set format (see data/golden_set.example.json):
    {"queries": [{"query": "...", "relevant": ["docs/path.md"]}]}
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.database import persistence
from app.services.kb.chunk import chunk_document
from app.services.kb.embeddings import build_embeddings
from app.services.kb.eval import evaluate, load_golden
from app.services.kb.parse import extract_pages
from app.services.kb.rerank import build_reranker
from app.services.kb.vectorstore import (
    InMemoryKbVectorStore,
    KbVectorStore,
    get_vector_store,
)


async def _resolve_kb(owner: str, kb: str) -> dict:
    """Find a KB by id or by name."""
    by_id = await persistence.kb.get_kb(owner, kb)
    if by_id is not None:
        return by_id
    for row in await persistence.kb.list_kbs(owner):
        if row["name"] == kb:
            return row
    raise SystemExit(f"KB {kb!r} not found for owner {owner!r}")


async def _ingest_into(store: KbVectorStore, owner: str, kb_id: str) -> int:
    """Parse + chunk + embed every document of the KB into the store."""
    docs = await persistence.kb.list_documents(owner, kb_id)
    embedded = 0
    for doc in docs:
        fetched = await persistence.kb.get_document_content(owner, doc["id"])
        if fetched is None:
            continue
        meta, data = fetched
        pages = extract_pages(meta["path"], data)
        chunks = chunk_document(
            meta["path"],
            pages,
            chunk_size=settings.kb_chunk_size,
            chunk_overlap=settings.kb_chunk_overlap,
        )
        if not chunks:
            continue
        # Live mode: drop stale vectors first (no unique constraint in Weaviate).
        store.delete_document(doc["id"])
        store.upsert(
            kb_id=kb_id,
            doc_id=doc["id"],
            owner=owner,
            path=meta["path"],
            chunks=chunks,
        )
        embedded += 1
    return embedded


async def _run(args: argparse.Namespace) -> None:
    await persistence.start()
    try:
        kb = await _resolve_kb(args.owner, args.kb)
        print(f"KB: {kb['name']} ({kb['id']}) — owner {args.owner}")
        if args.live:
            store = get_vector_store()
            if store is None:
                raise SystemExit("--live requires WEAVIATE_URL set and reachable")
            print("Store: Weaviate (live)")
        else:
            store = InMemoryKbVectorStore(embeddings=build_embeddings())
            print(f"Store: in-memory ({type(store._embeddings).__name__})")
        embedded = await _ingest_into(store, args.owner, kb["id"])
        print(f"Documents embedded: {embedded}")

        queries = load_golden(args.golden)
        print(f"Golden queries: {len(queries)} (limit={args.limit})")
        reranker = build_reranker() if args.rerank else None
        print("Rerank: " + (f"on ({settings.kb_rerank_model})" if reranker else "off"))
        print(f"{'alpha':>6}  {'recall@k':>9}  {'mrr':>7}  {'ndcg@k':>7}")
        best = None
        for alpha in args.alphas:
            result = evaluate(
                store,
                queries,
                owner=args.owner,
                alpha=alpha,
                limit=args.limit,
                reranker=reranker,
            )
            print(
                f"{alpha:>6.2f}  {result['recall_at_k']:>9.3f}  "
                f"{result['mrr']:>7.3f}  {result['ndcg_at_k']:>7.3f}"
            )
            if best is None or result["ndcg_at_k"] > best["ndcg_at_k"]:
                best = result
            if args.verbose:
                for q in result["per_query"]:
                    print(f"    {q.query!r} -> {q.top_hits}")
        print(f"Best alpha: {best['alpha']:.2f} (ndcg@{args.limit} {best['ndcg_at_k']:.3f})")
    finally:
        await persistence.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kb", required=True, help="KB id or name to evaluate")
    parser.add_argument("--owner", default="admin", help="owner of the KB (default: admin)")
    parser.add_argument(
        "--golden", required=True, help="golden set JSON (see data/golden_set.example.json)"
    )
    parser.add_argument(
        "--alphas", default="0,0.25,0.5,0.75,1.0", help="comma-separated alpha sweep"
    )
    parser.add_argument("--limit", type=int, default=5, help="top-k hits per query")
    parser.add_argument("--rerank", action="store_true", help="enable reranking (KB_RERANK_MODEL)")
    parser.add_argument("--live", action="store_true", help="use the configured Weaviate store")
    parser.add_argument("--verbose", action="store_true", help="print per-query hit lists")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        parser.error("--alphas must contain at least one value")
    args.alphas = alphas

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
