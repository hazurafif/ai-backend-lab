"""Live reranker test: real FlashRank cross-encoder vs the vector store ranking.

Standalone (no server, no Postgres, no API key needed): seeds a small demo
runbook corpus with the deterministic local embedder, then compares plain
hybrid retrieval against retrieve-broad/rerank-fine on a few sample queries.
First run downloads the flashrank model (~22 MB) to /tmp.

Usage:
    uv run python scripts/test_reranker.py
    KB_RERANK_MODEL=ms-marco-TinyBERT-L-2-v2 uv run python scripts/test_reranker.py

Expected output: per-query top-3 for the store ranking and the reranked
ranking (FlashRank order + scores), plus a note when they differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.services.kb.embeddings import LocalEmbeddings
from app.services.kb.rerank import FlashRankReranker, search_with_rerank
from app.services.kb.vectorstore import InMemoryKbVectorStore

CORPUS = [
    (
        "runbook/deploy.md",
        "To deploy the service, run kubectl rollout restart deployment app. "
        "The rollout command triggers a rolling update of the pods.",
    ),
    (
        "runbook/db-down.md",
        "When the database is down, check the connection pool and restart the "
        "postgres pod with kubectl rollout restart. Also verify the PVC is mounted.",
    ),
    (
        "runbook/backup.md",
        "Backups run nightly at 2am and are stored in S3. Restore with the "
        "restore script; retention is 30 days.",
    ),
    (
        "runbook/monitoring.md",
        "Grafana dashboards show latency and error rate. Alerts page the "
        "on-call engineer via pagerduty.",
    ),
    (
        "runbook/secrets.md",
        "Rotate secrets with the vault CLI. Store new values in the vault "
        "before updating the deployment manifest.",
    ),
]

QUERIES = [
    "how do I restart the application",
    "database is down what do I do",
    "where do backups go",
    "rotate the secret key",
]


def main() -> None:
    store = InMemoryKbVectorStore(embeddings=LocalEmbeddings())
    for index, (path, text) in enumerate(CORPUS):
        store.upsert(kb_id="demo", doc_id=f"d{index}", owner="demo", path=path, chunks=[text])

    model = settings.kb_rerank_model or "ms-marco-MiniLM-L-12-v2"
    print(f"Reranker: FlashRank {model} (first run downloads ~22MB to /tmp)")
    reranker = FlashRankReranker(model=model)

    for query in QUERIES:
        store_top = store.search(query, owner="demo", limit=3, alpha=0.5)
        reranked = search_with_rerank(store, reranker, query, owner="demo", limit=3, alpha=0.5)
        store_paths = [h.path for h in store_top]
        rerank_paths = [h.path for h in reranked]
        changed = "  <-- CHANGED" if store_paths != rerank_paths else ""
        print(f"\nQ: {query!r}{changed}")
        print("  store : " + " | ".join(store_paths))
        print("  rerank: " + " | ".join(rerank_paths))
        for hit in reranked:
            print(f"    {hit.path}  hybrid_score={hit.score:.3f}")


if __name__ == "__main__":
    main()
