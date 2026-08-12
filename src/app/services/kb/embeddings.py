"""Embeddings factory for the knowledge base pipeline.

- `OPENAI_API_KEY` present -> `OpenAIEmbeddings` (model + optional custom
  base URL from settings; default `text-embedding-3-small`).
- Otherwise a deterministic local embedder (hash-based bag of words) so the
  whole pipeline runs offline in dev/tests without any API key. Not meant for
  production retrieval quality.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re

from langchain_core.embeddings import Embeddings

from ...core.config import settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


class LocalEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words embedder (dev/tests only).

    Each token is hashed into a fixed-dimension one-hot bucket, then the
    vector is L2-normalized. Equal inputs always produce equal vectors, which
    keeps offline tests stable.
    """

    dim = 384

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _WORD_RE.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embeddings() -> Embeddings:
    """Real OpenAI embeddings when a key is configured, else the local one."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict = {
            "model": settings.embeddings_model,
            "base_url": settings.embeddings_base_url or None,
            "check_embedding_ctx_length": False,
        }
        if settings.embeddings_dimensions is not None:
            kwargs["dimensions"] = settings.embeddings_dimensions
        return OpenAIEmbeddings(**kwargs)
    logger.warning(
        "OPENAI_API_KEY not set: using the deterministic local embedder for the "
        "knowledge base (dev/tests only). Set OPENAI_API_KEY for real retrieval."
    )
    return LocalEmbeddings()
