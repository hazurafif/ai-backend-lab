"""Offline tests for the embeddings factory: MLX/Qwen3-Embedding path.

Verifies:

  - EMBEDDINGS_MLX_URL (env fallback) resolves to OpenAIEmbeddings against a
    local mlx_lm.server serving Qwen3-Embedding-0.6B, winning over
    OPENAI_API_KEY
  - InstructionAwareEmbeddings prefixes queries (not documents) with Qwen's
    "Instruct: ..." instruction for qwen3-embedding models
  - explicit EMBEDDINGS_QUERY_INSTRUCTION override and "" disable
  - missing connection + no env fallback -> deterministic local embedder
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

MLX_URL = "http://127.0.0.1:8080/v1"
MLX_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


def test_mlx_env_beats_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection_service, "resolved_embeddings", lambda: None)
    monkeypatch.setattr(embeddings_module.settings, "embeddings_mlx_url", MLX_URL)
    monkeypatch.setattr(embeddings_module.settings, "embeddings_mlx_model", MLX_MODEL)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")  # must lose to MLX

    embeddings = build_embeddings()

    assert isinstance(embeddings, InstructionAwareEmbeddings)
    inner = embeddings._inner
    assert isinstance(inner, OpenAIEmbeddings)
    assert inner.model == MLX_MODEL
    assert inner.openai_api_base == MLX_URL
    assert inner.openai_api_key.get_secret_value() == "mlx-local"


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


def test_no_mlx_and_no_openai_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection_service, "resolved_embeddings", lambda: None)
    monkeypatch.setattr(embeddings_module.settings, "embeddings_mlx_url", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert isinstance(build_embeddings(), LocalEmbeddings)
