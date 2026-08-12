"""Document ingestion pipeline: parse -> chunk -> embed -> vector store.

Runs the CPU/blocking stages (parsing, chunking, embedding, Weaviate calls)
in a threadpool so the event loop stays responsive. Document status is
`processing` while running, `ready` on success, `failed` + error on failure
(the exception is re-raised so the upload route can report per-file results).
"""

from __future__ import annotations

import logging

from fastapi.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.database import persistence
from .chunk import chunk_document
from .parse import extract_pages
from .vectorstore import KbUnavailableError, KbVectorStore, get_vector_store

logger = logging.getLogger(__name__)


async def ingest_document(doc: dict, data: bytes, vector_store: KbVectorStore | None = None) -> int:
    """Ingest one document's bytes into the vector store.

    Args:
        doc: document metadata row (id, kb_id, owner, path, ...).
        data: raw file bytes.
        vector_store: store to use; defaults to the configured singleton.

    Returns:
        The number of chunks stored.

    Raises:
        KbUnavailableError: vector store not configured/unreachable.
        Exception: parse/chunk failures (status is set to `failed` first).
    """
    owner, doc_id = doc["owner"], doc["id"]
    vs = vector_store or get_vector_store()
    if vs is None:
        raise KbUnavailableError("Vector store is not configured (set WEAVIATE_URL)")
    await persistence.kb.update_document(owner, doc_id, status="processing")
    try:
        pages = await run_in_threadpool(extract_pages, doc["path"], data)
        chunks = await run_in_threadpool(
            chunk_document,
            doc["path"],
            pages,
            chunk_size=settings.kb_chunk_size,
            chunk_overlap=settings.kb_chunk_overlap,
        )
        if not chunks:
            raise ValueError("No extractable text found in the file")
        count = await run_in_threadpool(
            vs.upsert,
            kb_id=doc["kb_id"],
            doc_id=doc_id,
            owner=owner,
            path=doc["path"],
            chunks=chunks,
        )
        await persistence.kb.update_document(
            owner, doc_id, status="ready", chunk_count=count, clear_error=True
        )
        logger.info("ingested %s: %d chunks", doc["path"], count)
        return count
    except Exception as exc:
        logger.exception("ingest failed for %s", doc["path"])
        await persistence.kb.update_document(
            owner, doc_id, status="failed", error=str(exc), chunk_count=0
        )
        raise


async def reindex_kb(kb_id: str, owner: str) -> dict:
    """Re-parse + re-embed every document of a KB (e.g. new embedding model)."""
    vector_store = get_vector_store()
    if vector_store is None:
        raise KbUnavailableError("Vector store is not configured (set WEAVIATE_URL)")
    docs = await persistence.kb.list_documents(owner, kb_id)
    processed = failed = 0
    for doc in docs:
        fetched = await persistence.kb.get_document_content(owner, doc["id"])
        if fetched is None:
            continue
        meta, data = fetched
        # Drop existing vectors first (no unique constraint on doc_id in Weaviate).
        await run_in_threadpool(vector_store.delete_document, doc["id"])
        try:
            await ingest_document(meta, data, vector_store)
            processed += 1
        except Exception:
            failed += 1
    return {"documents": len(docs), "processed": processed, "failed": failed}
