"""Embeddings factory for the knowledge base pipeline.

Resolution order in `build_embeddings()`:

1. Saved `embeddings` connection (base URL + API token, see /connections) ->
   OpenAIEmbeddings, any OpenAI-compatible endpoint.
2. `EMBEDDINGS_MLX_URL` set (env fallback enabled) -> OpenAIEmbeddings against
   a local `mlx_lm.server --embedding-model` process serving
   Qwen3-Embedding-0.6B on Apple Silicon (see scripts/mlx_embeddings.sh).
3. `OPENAI_API_KEY` present -> OpenAIEmbeddings (default
   `text-embedding-3-small`).
4. Otherwise a deterministic local embedder (hash-based bag of words) so the
   whole pipeline runs offline in dev/tests without any API key. Not meant for
   production retrieval quality.

Qwen3-Embedding models are instruction-aware: `embed_query()` gets an
"Instruct: ..." prefix (1-5% retrieval gain), passages are embedded bare.
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

# Qwen-recommended instruction for asymmetric retrieval: prepended to queries
# (not passages) for Qwen3-Embedding models.
DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query"
)


class InstructionAwareEmbeddings(Embeddings):
    """Wrap an embedder, prefixing an instruction to queries but not documents.

    Qwen3-Embedding is instruction-aware; Qwen's guidance is to prepend an
    instruction to queries (documents stay bare) for a ~1-5% retrieval gain.
    """

    def __init__(self, inner: Embeddings, instruction: str) -> None:
        self._inner = inner
        self._instruction = instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"{self._instruction}\n{text}")


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


def _query_instruction(model: str) -> str | None:
    """Instruction to prefix queries with, or None for plain embedding."""
    explicit = settings.embeddings_query_instruction
    if explicit == "":  # explicitly disabled
        return None
    if explicit is not None:
        return explicit
    if "qwen3-embedding" in model.lower():
        return DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION
    return None


def build_embeddings() -> Embeddings:
    """Build the KB embedder: saved connection > MLX > OpenAI env > local.

    Credentials resolve from the saved `embeddings` connection (base_url +
    api_token, see /connections) first. With no connection, .env keys are only
    consulted when the env fallback is opted in (CONNECTION_FALLBACK_ENV=true /
    PUT /settings connections.fallback_env); EMBEDDINGS_MLX_URL then wins over
    OPENAI_API_KEY so local Qwen3-Embedding-0.6B serving (scripts/
    mlx_embeddings.sh) is the default local path.
    """
    from ...services import settings as runtime_settings
    from ...services.connections import resolved_embeddings

    conn = resolved_embeddings()
    fallback_env = runtime_settings.connection_fallback_env()
    api_key: str | None = None
    base_url: str | None = None
    model = settings.embeddings_model
    if conn is None and fallback_env:
        if settings.embeddings_mlx_url:
            # mlx_lm.server ignores auth; the OpenAI client still wants a key.
            api_key = "mlx-local"
            base_url = settings.embeddings_mlx_url
            model = settings.embeddings_mlx_model
        elif os.environ.get("OPENAI_API_KEY"):
            api_key = os.environ["OPENAI_API_KEY"]
            base_url = settings.embeddings_base_url
    elif conn is not None:
        api_key = conn.get("api_token")
        base_url = conn.get("base_url") or settings.embeddings_base_url
    if api_key:
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url or None,
            "check_embedding_ctx_length": False,
        }
        if settings.embeddings_dimensions is not None:
            kwargs["dimensions"] = settings.embeddings_dimensions
        embeddings: Embeddings = OpenAIEmbeddings(**kwargs)
        instruction = _query_instruction(model)
        if instruction:
            embeddings = InstructionAwareEmbeddings(embeddings, instruction)
        return embeddings
    logger.warning(
        "No embeddings connection, EMBEDDINGS_MLX_URL, or OPENAI_API_KEY set: "
        "using the deterministic local embedder for the knowledge base "
        "(dev/tests only). Save an 'embeddings' connection via POST /connections "
        "or run scripts/mlx_embeddings.sh + set EMBEDDINGS_MLX_URL for real "
        "retrieval."
    )
    return LocalEmbeddings()
