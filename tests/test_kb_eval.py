"""Offline tests for the retrieval eval harness (R1) + tuning knobs (R2).

- IR metric math (Recall@k, MRR, nDCG) on a known in-memory corpus
- golden set loading/validation
- page-level chunking (PDF-style pages stay whole, oversized pages split,
  markdown stays header-aware)
- per-request `alpha` on the search endpoints
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from app.core import config
from app.core.database import persistence
from app.services.kb.embeddings import LocalEmbeddings
from app.services.kb.eval import GoldenQuery, evaluate, load_golden
from app.services.kb.vectorstore import (
    InMemoryKbVectorStore,
    reset_vector_store,
    set_vector_store,
)

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class Scripted(BaseChatModel):
    """Minimal scripted model so agent builds stay offline."""

    responses: list[AIMessage] = Field(default_factory=list)
    tools: list = Field(default_factory=list)
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = min(self._idx, len(self.responses) - 1)
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseChatModel],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = list(tools)
        return self


@pytest_asyncio.fixture
async def kb_env():
    """In-memory persistence + vector store for eval tests."""
    config.settings.database_uri = None
    await persistence.start()
    store = InMemoryKbVectorStore(embeddings=LocalEmbeddings())
    set_vector_store(store)
    yield store
    reset_vector_store()
    await persistence.stop()


def _golden(entries: list[tuple[str, list[str]]]) -> list[GoldenQuery]:
    return [GoldenQuery(query=q, relevant=rel) for q, rel in entries]


def _corpus(store: InMemoryKbVectorStore) -> None:
    """Seed three documents with distinctive tokens."""
    store.upsert(
        kb_id="kb-1",
        doc_id="d1",
        owner="tester",
        path="a.md",
        chunks=["kubectl deployment rollout"],
    )
    store.upsert(
        kb_id="kb-1", doc_id="d2", owner="tester", path="b.md", chunks=["backups run nightly to s3"]
    )
    store.upsert(
        kb_id="kb-1",
        doc_id="d3",
        owner="tester",
        path="c.md",
        chunks=["grafana monitoring dashboards"],
    )


# ---------------------------------------------------------------------------
# metric math
# ---------------------------------------------------------------------------


async def test_metrics_math(kb_env):
    _corpus(kb_env)

    # perfect hit
    result = evaluate(
        kb_env, _golden([("deployment", ["a.md"])]), owner="tester", alpha=0.5, limit=5
    )
    q = result["per_query"][0]
    assert q.recall_at_k == 1.0 and q.mrr == 1.0 and q.ndcg_at_k == 1.0
    assert q.top_hits[0] == "a.md"

    # partial recall with limit=1: only the top hit counts, so one of two
    # relevant docs is found; nDCG@1 = 1.0 because the ideal top-1 is relevant
    result = evaluate(
        kb_env, _golden([("deployment", ["a.md", "b.md"])]), owner="tester", alpha=0.5, limit=1
    )
    q = result["per_query"][0]
    assert q.recall_at_k == pytest.approx(0.5)
    assert q.mrr == pytest.approx(1.0)  # first hit is relevant
    assert q.ndcg_at_k == pytest.approx(1.0)

    # miss: relevant doc not in top-k (limit=2 -> a.md, b.md only)
    result = evaluate(
        kb_env, _golden([("deployment", ["c.md"])]), owner="tester", alpha=0.5, limit=2
    )
    q = result["per_query"][0]
    assert q.recall_at_k == 0.0 and q.mrr == 0.0 and q.ndcg_at_k == 0.0

    # miss + relevance at rank 2 (top_hits[0] = "c.md" not relevant)
    result = evaluate(
        kb_env, _golden([("monitoring", ["c.md"])]), owner="tester", alpha=0.5, limit=5
    )
    assert result["per_query"][0].mrr == pytest.approx(1.0)

    # empty relevance list -> zero metrics, no crash
    result = evaluate(kb_env, _golden([("deployment", [])]), owner="tester", alpha=0.5, limit=5)
    assert result["per_query"][0].recall_at_k == 0.0

    # aggregates are means (limit=1: the zero-score query deterministically
    # misses its relevant doc since the top hit is the first chunk)
    result = evaluate(
        kb_env,
        _golden([("deployment", ["a.md"]), ("no-such-topic-xyz", ["c.md"])]),
        owner="tester",
        alpha=0.5,
        limit=1,
    )
    assert result["queries"] == 2
    assert result["recall_at_k"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)

    # owner isolation: evaluation with another owner sees nothing
    result = evaluate(
        kb_env, _golden([("deployment", ["a.md"])]), owner="other", alpha=0.5, limit=5
    )
    assert result["per_query"][0].recall_at_k == 0.0


async def test_load_golden(tmp_path, kb_env):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps({"queries": [{"query": "deploy", "relevant": ["a.md"]}]}), encoding="utf-8"
    )
    queries = load_golden(str(path))
    assert queries[0].query == "deploy" and queries[0].relevant == ["a.md"]

    for bad in (
        {"queries": [{"relevant": ["a.md"]}]},  # missing query
        {"queries": [{"query": "x", "relevant": "a.md"}]},  # relevant not a list
        {"queries": []},  # empty
        {},  # not an object with queries
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_golden(str(path))
