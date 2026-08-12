"""Vector store for KB chunks: Weaviate in production, in-memory for tests.

Two implementations of the same protocol:

- `WeaviateKbVectorStore` — hybrid search (BM25F + vector, `alpha` blend)
  over a single `KnowledgeChunk` collection, filtered by `owner` (always)
  and optionally `kb_id`. Vectors are computed in app code (no Weaviate
  vectorizer modules) so the embedding provider is swappable and offline
  tests never touch the network.
- `InMemoryKbVectorStore` — brute-force cosine + token overlap for offline
  tests (same protocol).

`get_vector_store()` returns the configured singleton (None when
`WEAVIATE_URL` is unset); `set_vector_store()` swaps it (tests, reconnect).
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit

from langchain_core.embeddings import Embeddings

from ...core.config import settings
from ...schema.kb_schema import SearchHit
from .embeddings import build_embeddings

logger = logging.getLogger(__name__)

COLLECTION = "KnowledgeChunk"

_WORD_RE = re.compile(r"\w+")


class KbUnavailableError(RuntimeError):
    """Vector store is not configured or unreachable."""


class KbVectorStore:
    """Protocol for KB chunk storage + hybrid retrieval."""

    def upsert(self, *, kb_id: str, doc_id: str, owner: str, path: str, chunks: list[str]) -> int:
        """Embed and store chunks of one document; returns the chunk count."""
        raise NotImplementedError

    def search(
        self,
        query: str,
        *,
        owner: str,
        kb_id: str | None = None,
        limit: int = 5,
        alpha: float = 0.5,
    ) -> list[SearchHit]:
        """Hybrid search over chunks the owner can see (owner filter is mandatory)."""
        raise NotImplementedError

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document; returns the number deleted."""
        raise NotImplementedError

    def delete_kb(self, kb_id: str) -> int:
        """Remove every chunk of a knowledge base; returns the number deleted."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class WeaviateKbVectorStore(KbVectorStore):
    """Weaviate-backed store. Connection is lazy (weaviate-client v4)."""

    def __init__(self, url: str, api_key: str | None, embeddings: Embeddings) -> None:
        import weaviate

        parts = urlsplit(url)
        self._client: Any = weaviate.connect_to_custom(
            http_host=parts.hostname or "localhost",
            http_port=parts.port or (443 if parts.scheme == "https" else 80),
            http_secure=parts.scheme == "https",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )
        self._embeddings = embeddings

    def _ensure_collection(self) -> Any:
        from weaviate.classes.config import Configure, DataType, Property

        if not self._client.collections.exists(COLLECTION):
            try:
                self._client.collections.create(
                    name=COLLECTION,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="owner", data_type=DataType.TEXT),
                        Property(name="kb_id", data_type=DataType.TEXT),
                        Property(name="doc_id", data_type=DataType.TEXT),
                        Property(name="path", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="content", data_type=DataType.TEXT),
                    ],
                )
                logger.info("Created Weaviate collection %s", COLLECTION)
            except Exception as exc:
                raise KbUnavailableError(f"Weaviate collection setup failed: {exc}") from exc
        return self._client.collections.get(COLLECTION)

    def upsert(self, *, kb_id: str, doc_id: str, owner: str, path: str, chunks: list[str]) -> int:
        try:
            collection = self._ensure_collection()
            vectors = self._embeddings.embed_documents(chunks)
            with collection.batch.fixed_size(batch_size=50) as batch:
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
                    batch.add_object(
                        properties={
                            "owner": owner,
                            "kb_id": kb_id,
                            "doc_id": doc_id,
                            "path": path,
                            "chunk_index": index,
                            "content": chunk,
                        },
                        vector=vector,
                    )
            return len(chunks)
        except KbUnavailableError:
            raise
        except Exception as exc:
            raise KbUnavailableError(f"Weaviate upsert failed: {exc}") from exc

    def search(
        self,
        query: str,
        *,
        owner: str,
        kb_id: str | None = None,
        limit: int = 5,
        alpha: float = 0.5,
    ) -> list[SearchHit]:
        from weaviate.classes.query import Filter, MetadataQuery

        try:
            collection = self._ensure_collection()
            filters = Filter.by_property("owner").equal(owner)
            if kb_id is not None:
                filters &= Filter.by_property("kb_id").equal(kb_id)
            query_vector = self._embeddings.embed_query(query)
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=filters,
                return_metadata=MetadataQuery(score=True),
            )
            hits: list[SearchHit] = []
            for obj in response.objects:
                props = obj.properties
                score = obj.metadata.score if obj.metadata is not None else 0.0
                hits.append(
                    SearchHit(
                        path=props["path"],
                        content=props["content"],
                        score=round(float(score), 4),
                        chunk_index=props["chunk_index"],
                        doc_id=props["doc_id"],
                    )
                )
            return hits
        except KbUnavailableError:
            raise
        except Exception as exc:
            raise KbUnavailableError(f"Weaviate search failed: {exc}") from exc

    def delete_document(self, doc_id: str) -> int:
        from weaviate.classes.query import Filter

        try:
            collection = self._ensure_collection()
            result = collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
            return len(result.objects)
        except KbUnavailableError:
            raise
        except Exception as exc:
            raise KbUnavailableError(f"Weaviate delete failed: {exc}") from exc

    def delete_kb(self, kb_id: str) -> int:
        from weaviate.classes.query import Filter

        try:
            collection = self._ensure_collection()
            result = collection.data.delete_many(where=Filter.by_property("kb_id").equal(kb_id))
            return len(result.objects)
        except KbUnavailableError:
            raise
        except Exception as exc:
            raise KbUnavailableError(f"Weaviate delete failed: {exc}") from exc

    def close(self) -> None:
        self._client.close()


class InMemoryKbVectorStore(KbVectorStore):
    """Brute-force hybrid store for offline tests (and dev without Weaviate).

    Score = alpha * cosine(query, chunk) + (1 - alpha) * query-token overlap,
    matching Weaviate's hybrid spirit well enough for scripted tests.
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._chunks: list[dict] = []  # {owner, kb_id, doc_id, path, index, content, emb}

    def upsert(self, *, kb_id: str, doc_id: str, owner: str, path: str, chunks: list[str]) -> int:
        vectors = self._embeddings.embed_documents(chunks)
        for index, (chunk, emb) in enumerate(zip(chunks, vectors, strict=False)):
            self._chunks.append(
                {
                    "owner": owner,
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "path": path,
                    "index": index,
                    "content": chunk,
                    "emb": emb,
                }
            )
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        owner: str,
        kb_id: str | None = None,
        limit: int = 5,
        alpha: float = 0.5,
    ) -> list[SearchHit]:
        query_vector = self._embeddings.embed_query(query)
        query_tokens = set(_WORD_RE.findall(query.lower()))
        scored: list[tuple[float, dict]] = []
        for chunk in self._chunks:
            if chunk["owner"] != owner:
                continue
            if kb_id is not None and chunk["kb_id"] != kb_id:
                continue
            cosine = _cosine(query_vector, chunk["emb"])
            overlap = 0.0
            if query_tokens:
                tokens = set(_WORD_RE.findall(chunk["content"].lower()))
                overlap = len(query_tokens & tokens) / len(query_tokens)
            scored.append((alpha * cosine + (1 - alpha) * overlap, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                path=chunk["path"],
                content=chunk["content"],
                score=round(score, 4),
                chunk_index=chunk["index"],
                doc_id=chunk["doc_id"],
            )
            for score, chunk in scored[:limit]
        ]

    def delete_document(self, doc_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c["doc_id"] != doc_id]
        return before - len(self._chunks)

    def delete_kb(self, kb_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c["kb_id"] != kb_id]
        return before - len(self._chunks)

    def close(self) -> None:
        self._chunks.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5 or 1.0
    norm_b = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# singleton wiring
# ---------------------------------------------------------------------------

_configured: KbVectorStore | None = None
_built = False


def build_vector_store() -> KbVectorStore | None:
    """Build the configured store (Weaviate when WEAVIATE_URL is set)."""
    if not settings.weaviate_url:
        return None
    try:
        return WeaviateKbVectorStore(
            url=settings.weaviate_url,
            api_key=settings.weaviate_api_key,
            embeddings=build_embeddings(),
        )
    except Exception:
        logger.exception("Failed to build Weaviate vector store; KB search disabled")
        return None


def get_vector_store() -> KbVectorStore | None:
    """The process-wide vector store, built once from settings."""
    global _configured, _built
    if not _built:
        _configured = build_vector_store()
        _built = True
    return _configured


def set_vector_store(store: KbVectorStore | None) -> None:
    """Replace the singleton (tests inject an in-memory store)."""
    global _configured, _built
    _configured = store
    _built = store is not None


def reset_vector_store() -> None:
    """Drop the singleton so the next get_vector_store() rebuilds from settings."""
    global _configured, _built
    _configured = None
    _built = False
