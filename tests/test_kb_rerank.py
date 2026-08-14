"""Offline tests for the reranking stage (R3): retrieve broad, rerank fine.

A deterministic FakeReranker replaces FlashRank in tests: it prefers hits
whose path contains a marker, so reordering is fully predictable. Verifies:

- search_with_rerank pipeline order + limit, identity = plain search
- REST search endpoints rerank (per-KB and global)
- the agent tool reranks before formatting
- the eval harness can measure rerank vs plain retrieval (the R3 gate)
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from app.core import config
from app.core.database import persistence
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream
from app.services.kb.embeddings import LocalEmbeddings
from app.services.kb.eval import GoldenQuery, evaluate
from app.services.kb.rerank import (
    IdentityReranker,
    get_reranker,
    reset_reranker,
    search_with_rerank,
    set_reranker,
)
from app.services.kb.tool import build_kb_search_tool
from app.services.kb.vectorstore import (
    InMemoryKbVectorStore,
    reset_vector_store,
    set_vector_store,
)

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class FakeReranker:
    """Deterministic reranker: marker hits first, then store score order."""

    def __init__(self, marker: str = "preferred") -> None:
        self._marker = marker
        self.calls = 0

    def rerank(self, query: str, hits: list, top_k: int) -> list:
        self.calls += 1
        ranked = sorted(hits, key=lambda hit: (self._marker not in hit.path, -hit.score))
        return ranked[:top_k]


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
async def memory_persistence():
    """In-memory checkpointer/store for direct agent tests."""
    config.settings.database_uri = None
    await persistence.start()
    yield persistence
    await persistence.stop()


@pytest_asyncio.fixture
async def rerank_env():
    """In-memory persistence + vector store + fake reranker."""
    config.settings.database_uri = None
    await persistence.start()
    store = InMemoryKbVectorStore(embeddings=LocalEmbeddings())
    set_vector_store(store)
    fake = FakeReranker()
    set_reranker(fake)
    yield store, fake
    reset_reranker()
    reset_vector_store()
    await persistence.stop()


def _corpus(store: InMemoryKbVectorStore) -> None:
    """a.md ranks first for 'deployment'; preferred.md is the reranker's pick."""
    store.upsert(
        kb_id="kb-1",
        doc_id="d1",
        owner="tester",
        path="a.md",
        chunks=["kubectl deployment rollout"],
    )
    store.upsert(
        kb_id="kb-1",
        doc_id="d2",
        owner="tester",
        path="preferred.md",
        chunks=["deployment notes about kubectl"],
    )
    store.upsert(
        kb_id="kb-1", doc_id="d3", owner="tester", path="b.md", chunks=["backups run nightly to s3"]
    )


def _store_order(store) -> list[str]:
    return [h.path for h in store.search("deployment", owner="tester", limit=5, alpha=0.5)]


# ---------------------------------------------------------------------------
# pipeline unit
# ---------------------------------------------------------------------------


async def test_search_with_rerank_pipeline(rerank_env):
    store, fake = rerank_env
    _corpus(store)

    # identity reranker = plain search (order unchanged, limit respected)
    hits = search_with_rerank(store, IdentityReranker(), "deployment", owner="tester", limit=1)
    assert [h.path for h in hits] == ["a.md"]

    # fake reranker reorders: preferred.md first, limit respected
    assert _store_order(store)[0] == "a.md"  # precondition: store disagrees
    hits = search_with_rerank(store, fake, "deployment", owner="tester", limit=1)
    assert [h.path for h in hits] == ["preferred.md"]
    assert fake.calls == 1
    hits = search_with_rerank(store, fake, "deployment", owner="tester", limit=2)
    assert [h.path for h in hits] == ["preferred.md", "a.md"]

    # kb_id filter still applies before reranking
    store.upsert(
        kb_id="kb-2",
        doc_id="d9",
        owner="tester",
        path="preferred.md",
        chunks=["deployment notes about kubectl"],
    )
    hits = search_with_rerank(store, fake, "deployment", owner="tester", kb_id="kb-1", limit=2)
    assert [h.path for h in hits] == ["preferred.md", "a.md"]


async def test_default_reranker_is_identity(rerank_env):
    reset_reranker()
    assert isinstance(get_reranker(), IdentityReranker)


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


async def test_search_endpoints_rerank(rerank_env):
    store, fake = rerank_env
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
        kb = (await http.post("/knowledge", json={"name": "rr"})).json()
        # seed the vector store under the API-created KB id
        for path, chunk in (
            ("a.md", "kubectl deployment rollout"),
            ("preferred.md", "deployment notes about kubectl"),
            ("b.md", "backups run nightly to s3"),
        ):
            store.upsert(kb_id=kb["id"], doc_id=path, owner="tester", path=path, chunks=[chunk])
        # per-KB search reranks
        r = await http.get(f"/knowledge/{kb['id']}/search", params={"q": "deployment", "limit": 2})
        assert r.status_code == 200, r.text
        assert [h["path"] for h in r.json()["hits"]] == ["preferred.md", "a.md"]
        # global search reranks too
        r = await http.get("/knowledge/search", params={"q": "deployment", "limit": 1})
        assert r.status_code == 200
        assert [h["path"] for h in r.json()["hits"]] == ["preferred.md"]
        assert fake.calls >= 2


# ---------------------------------------------------------------------------
# agent tool
# ---------------------------------------------------------------------------


async def test_agent_tool_reranks(rerank_env, memory_persistence):
    store, _ = rerank_env
    _corpus(store)

    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="search_knowledge_base",
                        args={"query": "deployment", "top_k": 2},
                    )
                ],
            ),
            AIMessage(content="The answer."),
        ]
    )
    agent = build_agent(
        checkpointer=memory_persistence.checkpointer,
        store=memory_persistence.store,
        extra_tools=[build_kb_search_tool(vector_store=store)],
        model=model,
        system_prompt="test",
    )
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, "tester", message="deployment?"):
        ev, _, rest = chunk.partition("\n")
        events.append((ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())))
    tool_end = next(
        d for e, d in events if e == "tool_end" and d["name"] == "search_knowledge_base"
    )
    output = tool_end["output"]["content"]
    assert "preferred.md" in output, output
    assert output.index("preferred.md") < output.index("a.md"), output


# ---------------------------------------------------------------------------
# eval gate: rerank vs plain retrieval on a golden set
# ---------------------------------------------------------------------------


async def test_eval_measures_rerank_impact(rerank_env):
    store, fake = rerank_env
    _corpus(store)
    golden = [GoldenQuery(query="deployment", relevant=["preferred.md"])]

    plain = evaluate(store, golden, owner="tester", alpha=0.5, limit=1)
    reranked = evaluate(store, golden, owner="tester", alpha=0.5, limit=1, reranker=fake)
    assert plain["per_query"][0].recall_at_k == 0.0  # store ranks a.md first
    assert reranked["per_query"][0].recall_at_k == 1.0  # reranker promotes preferred.md
    assert reranked["rerank"] is True and plain["rerank"] is False
