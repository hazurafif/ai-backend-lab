"""Offline tests for the embeddings factory.

Verifies:

  - a saved `embeddings` connection resolves to OpenAIEmbeddings against the
    connection's base URL (no env involved)
  - InstructionAwareEmbeddings prefixes queries (not documents) with Qwen's
    "Instruct: ..." instruction for qwen3-embedding models
  - explicit EMBEDDINGS_QUERY_INSTRUCTION override and "" disable
  - no connection -> deterministic local embedder (no env fallback exists)
"""

from __future__ import annotations

import pytest
from langchain_openai import OpenAIEmbeddings

from app.services import connections as connection_service
from app.services.kb import embeddings as embeddings_module
from app.services.kb.embeddings import (
    DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION,
    InstructionAwareEmbeddings,
    LocalEmbeddings,
    _query_instruction,
    build_embeddings,
)

MLX_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


def test_instruction_wrapper_prefixes_queries_only() -> None:
    inner = LocalEmbeddings()
    wrapped = InstructionAwareEmbeddings(inner, DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION)
    text = "sushi in SF"

    assert wrapped.embed_documents([text]) == inner.embed_documents([text])
    assert wrapped.embed_query(text) == inner.embed_query(
        f"{DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION}\n{text}"
    )


def test_query_instruction_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings_module.settings, "embeddings_query_instruction", None)
    assert _query_instruction(MLX_MODEL) == DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION
    assert _query_instruction("text-embedding-3-small") is None


def test_query_instruction_override_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embeddings_module.settings, "embeddings_query_instruction", "Instruct: custom"
    )
    assert _query_instruction("text-embedding-3-small") == "Instruct: custom"
    monkeypatch.setattr(embeddings_module.settings, "embeddings_query_instruction", "")
    assert _query_instruction(MLX_MODEL) is None


def test_saved_connection_resolves_openai_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        connection_service,
        "resolved_embeddings",
        lambda: {
            "base_url": "http://localhost:9999/emb",
            "api_token": "sk-emb-token",
        },
    )

    embeddings = build_embeddings()

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-small"
    assert embeddings.openai_api_base == "http://localhost:9999/emb"
    assert embeddings.openai_api_key.get_secret_value() == "sk-emb-token"


def test_no_connection_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """No connection + env keys present -> deterministic local embedder.

    There is no env fallback: OPENAI_API_KEY / EMBEDDINGS_MLX_URL are never
    consulted, so the local embedder is the only offline path.
    """
    monkeypatch.setattr(connection_service, "resolved_embeddings", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")  # gitguardian:ignore

    assert isinstance(build_embeddings(), LocalEmbeddings)
