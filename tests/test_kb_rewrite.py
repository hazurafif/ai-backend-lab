"""Offline tests for query rewriting (R4).

A deterministic FakeRewriter maps queries to fixed rewritten forms, so the
pipeline order is fully predictable: the original query finds nothing useful
(store returns zero-score chunks in insertion order), the rewritten query
promotes the right document. Verifies:

- search_with_rerank uses the rewritten query (REST + agent tool paths)
- SearchOut.query still carries the user's original query
- identity rewriter is the default; LLM rewriter's trivial-query gate never
  touches the LLM for short/single-token queries
- the eval harness can measure rewrite impact
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from app.core import config
from app.core.database import persistence
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.kb.embeddings import LocalEmbeddings
from app.services.kb.eval import GoldenQuery, evaluate
from app.services.kb.rerank import search_with_rerank
from app.services.kb.rewrite import (
    IdentityQueryRewriter,
    LLMQueryRewriter,
    get_rewriter,
    reset_rewriter,
    set_rewriter,
)
from app.services.kb.vectorstore import (
    InMemoryKbVectorStore,
    reset_vector_store,
    set_vector_store,
)

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class FakeRewriter:
    """Deterministic rewriter: fixed mapping, unknown queries unchanged."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls = 0

    def rewrite(self, query: str) -> str:
        self.calls += 1
        return self._mapping.get(query, query)


class Scripted(BaseChatModel):
    """Scripted model so agent builds stay offline."""

    responses: list[AIMessage] = Field(default_factory=list)
    tools: list = Field(default_factory=list)
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages,
        stop=None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        i = min(self._idx, len(self.responses) - 1)
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:
        self.tools = list(tools)
        return self


@pytest_asyncio.fixture
async def rewrite_env():
    """In-memory persistence + vector store + fake rewriter (zzz -> deployment)."""
    config.settings.database_uri = None
    await persistence.start()
    store = InMemoryKbVectorStore(embeddings=LocalEmbeddings())
    set_vector_store(store)
    fake = FakeRewriter({"zzz": "deployment"})
    set_rewriter(fake)
    yield store, fake
    reset_rewriter()
    reset_vector_store()
    await persistence.stop()


def _corpus(store: InMemoryKbVectorStore) -> None:
    """x.md first (so zero-score queries hit it); deploy.md is the rewrite target."""
    store.upsert(
        kb_id="kb-1", doc_id="d1", owner="tester", path="x.md", chunks=["backups run nightly to s3"]
    )
    store.upsert(
        kb_id="kb-1",
        doc_id="d2",
        owner="tester",
        path="deploy.md",
        chunks=["kubectl deployment rollout"],
    )


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


async def test_search_with_rewrite(rewrite_env):
    store, fake = rewrite_env
    _corpus(store)

    # without a rewriter the zero-score query returns insertion order (x.md first)
    plain = search_with_rerank(store, None, "zzz", owner="tester", limit=1)
    assert plain and plain[0].path == "x.md"

    # with the rewriter the rewritten query promotes deploy.md
    hits = search_with_rerank(store, None, "zzz", owner="tester", limit=1, rewriter=fake)
    assert [h.path for h in hits] == ["deploy.md"]
    assert fake.calls == 1

    # unknown queries pass through unchanged
    hits = search_with_rerank(store, None, "deployment", owner="tester", limit=1, rewriter=fake)
    assert [h.path for h in hits] == ["deploy.md"]


async def test_rewrite_composes_with_rerank(rewrite_env):
    store, fake_rewriter = rewrite_env
    _corpus(store)
    calls = {"rerank": 0}

    class OrderReranker:
        def rerank(self, query, hits, top_k):
            calls["rerank"] += 1
            return hits[:top_k]

    hits = search_with_rerank(
        store, OrderReranker(), "zzz", owner="tester", limit=1, rewriter=fake_rewriter
    )
    assert [h.path for h in hits] == ["deploy.md"]  # rewrite -> retrieve -> rerank
    assert calls["rerank"] == 1


async def test_default_rewriter_is_identity(rewrite_env):
    reset_rewriter()
    assert isinstance(get_rewriter(), IdentityQueryRewriter)


async def test_llm_rewriter_gate_never_touches_llm():
    """Trivial queries return as-is without constructing the LLM."""
    rewriter = LLMQueryRewriter(model="this-model-does-not-exist", min_length=8)
    assert rewriter.rewrite("hi") == "hi"  # too short
    assert rewriter.rewrite("kubectl") == "kubectl"  # single token
    assert rewriter.rewrite("  ") == ""  # blank


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


async def _client(app) -> httpx.AsyncClient:
    await persistence.users.create_user(username="tester", hashed_password="x")
    token = create_access_token(data={"sub": "tester"})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_search_endpoints_rewrite(rewrite_env):
    store, fake = rewrite_env
    _corpus(store)
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=Scripted(responses=[AIMessage(content="ok")]),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client(app) as http:
        kb = (await http.post("/knowledge", json={"name": "rw"})).json()
        for path, chunk in (
            ("x.md", "backups run nightly to s3"),
            ("deploy.md", "kubectl deployment rollout"),
        ):
            store.upsert(kb_id=kb["id"], doc_id=path, owner="tester", path=path, chunks=[chunk])

        # per-KB search rewrites; the response keeps the original query
        r = await http.get(f"/knowledge/{kb['id']}/search", params={"q": "zzz", "limit": 1})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["query"] == "zzz"
        assert [h["path"] for h in body["hits"]] == ["deploy.md"]

        # global search rewrites too
        r = await http.get("/knowledge/search", params={"q": "zzz", "limit": 1})
        assert r.status_code == 200
        assert [h["path"] for h in r.json()["hits"]] == ["deploy.md"]
        assert fake.calls >= 2


# ---------------------------------------------------------------------------
# eval gate
# ---------------------------------------------------------------------------


async def test_eval_measures_rewrite_impact(rewrite_env):
    store, fake = rewrite_env
    _corpus(store)
    golden = [GoldenQuery(query="zzz", relevant=["deploy.md"])]

    plain = evaluate(store, golden, owner="tester", alpha=0.5, limit=1)
    rewritten = evaluate(store, golden, owner="tester", alpha=0.5, limit=1, rewriter=fake)
    assert plain["per_query"][0].recall_at_k == 0.0  # zero-score query hits x.md
    assert rewritten["per_query"][0].recall_at_k == 1.0  # rewrite promotes deploy.md
    assert rewritten["rewrite"] is True and plain["rewrite"] is False
