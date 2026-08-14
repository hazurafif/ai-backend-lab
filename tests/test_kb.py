"""Offline tests for the knowledge base feature (upload, ingest, search, agent tool).

No network, no API keys, no Postgres: the KbStore falls back to in-memory
dicts and the vector store is `InMemoryKbVectorStore` with the deterministic
`LocalEmbeddings`. Verifies:

  - /kb CRUD + owner isolation (user B cannot see/touch user A's KBs)
  - multipart upload (files + relative paths = folder upload), per-file
    validation (extension, path traversal, size), document status lifecycle
  - hybrid search endpoint + vector cleanup on document/KB deletion
  - the agent's `search_knowledge_base` tool resolves the runtime user and
    only sees that user's chunks (through the full /chat SSE pipeline)
"""

from __future__ import annotations

import io
import json
import textwrap
import zipfile
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
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
from app.services.kb.chunk import chunk_document
from app.services.kb.embeddings import LocalEmbeddings
from app.services.kb.parse import extract_pages
from app.services.kb.tool import build_kb_search_tool
from app.services.kb.vectorstore import (
    InMemoryKbVectorStore,
    reset_vector_store,
    set_vector_store,
)

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class Scripted(BaseChatModel):
    """Returns a scripted sequence of AIMessages, clamping at the last."""

    responses: list[AIMessage] = Field(default_factory=list)
    tools: Sequence[dict | type] = ()
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
        self.tools = tools
        return self


@pytest_asyncio.fixture
async def memory_persistence():
    """Force in-memory checkpointer/store and start the persistence singleton."""
    config.settings.database_uri = None
    await persistence.start()
    yield persistence
    await persistence.stop()


@pytest_asyncio.fixture
async def kb_env(memory_persistence):
    """In-memory persistence + an in-memory vector store with local embeddings."""
    store = InMemoryKbVectorStore(embeddings=LocalEmbeddings())
    set_vector_store(store)
    yield store
    reset_vector_store()


async def _client_for(app, username: str = "tester", role: str = "user") -> httpx.AsyncClient:
    await persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def _scripted_model() -> Scripted:
    return Scripted(responses=[AIMessage(content="Final answer from the agent.")])


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


async def collect_stream(agent, username, **kwargs) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, username, **kwargs):
        events.append(parse_sse_chunk(chunk))
    return events


# ---------------------------------------------------------------------------
# KB CRUD + isolation
# ---------------------------------------------------------------------------


async def test_kb_crud_and_isolation(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        # create
        r = await http.post(
            "/knowledge", json={"name": "Engineering Docs", "description": "Runbook"}
        )
        assert r.status_code == 201, r.text
        kb = r.json()
        assert kb["name"] == "Engineering Docs"
        assert kb["document_count"] == 0

        # legacy /kb alias answers identically
        r_alias = await http.post("/kb", json={"name": "Alias KB"})
        assert r_alias.status_code == 201, r_alias.text
        r_alias_get = await http.get(f"/kb/{r_alias.json()['id']}")
        assert r_alias_get.status_code == 200, r_alias_get.text

        # duplicate name -> 409
        r = await http.post("/knowledge", json={"name": "Engineering Docs"})
        assert r.status_code == 409, r.text

        # invalid name -> 422
        r = await http.post("/knowledge", json={"name": "../evil"})
        assert r.status_code == 422, r.text

        # list + get (both the /knowledge and legacy /kb KBs appear)
        r = await http.get("/knowledge")
        ids = [k["id"] for k in r.json()]
        assert r.status_code == 200 and kb["id"] in ids and r_alias.json()["id"] in ids
        r = await http.get(f"/knowledge/{kb['id']}")
        assert r.status_code == 200 and r.json()["name"] == "Engineering Docs"

        # patch
        r = await http.patch(f"/knowledge/{kb['id']}", json={"name": "Docs", "description": None})
        assert r.status_code == 200 and r.json()["name"] == "Docs"

        # isolation: another user sees nothing and cannot touch it
        async with await _client_for(app, username="other") as other:
            r = await other.get("/knowledge")
            assert r.status_code == 200 and r.json() == []
            r = await other.get(f"/knowledge/{kb['id']}")
            assert r.status_code == 404, r.text
            r = await other.delete(f"/knowledge/{kb['id']}")
            assert r.status_code == 404, r.text

        # delete
        r = await http.delete(f"/knowledge/{kb['id']}")
        assert r.status_code == 204
        r = await http.get(f"/knowledge/{kb['id']}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# upload + ingest + search
# ---------------------------------------------------------------------------


async def test_upload_folder_and_search(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "runbook"})).json()
        kb_id = kb["id"]

        # folder upload: two files with relative paths + a markdown with headers
        files = [
            (
                "files",
                (
                    "guides/deploy.md",
                    b"# Deployment\n\nThe service is deployed with kubectl.",
                    "text/markdown",
                ),
            ),
            ("files", ("guides/backup.md", b"Backups run every night to S3.", "text/markdown")),
        ]
        paths = {"paths": ["guides/deploy.md", "guides/backup.md"]}
        r = await http.post(f"/knowledge/{kb_id}/files", files=files, data=paths)
        assert r.status_code == 200, r.text
        body = r.json()
        assert all(res["ok"] for res in body["results"]), body
        assert {res["path"] for res in body["results"]} == {
            "guides/deploy.md",
            "guides/backup.md",
        }

        # documents listed with status ready
        r = await http.get(f"/knowledge/{kb_id}/files")
        docs = r.json()
        assert len(docs) == 2
        assert {d["path"] for d in docs} == {"guides/deploy.md", "guides/backup.md"}
        assert all(d["status"] == "ready" for d in docs)
        assert all(d["chunk_count"] >= 1 for d in docs)

        # KB stats reflect documents + chunks
        r = await http.get(f"/knowledge/{kb_id}")
        assert r.json()["document_count"] == 2
        assert r.json()["chunk_count"] >= 2

        # hybrid search finds the deployment passage
        r = await http.get(
            f"/knowledge/{kb_id}/search", params={"q": "kubectl deployment", "limit": 5}
        )
        assert r.status_code == 200, r.text
        hits = r.json()["hits"]
        assert hits, "expected search hits"
        assert hits[0]["path"] == "guides/deploy.md"
        assert "kubectl" in hits[0]["content"]

        # duplicate path -> per-file error result
        r = await http.post(
            f"/knowledge/{kb_id}/files",
            files=[("files", ("guides/deploy.md", b"dup", "text/markdown"))],
            data={"paths": "guides/deploy.md"},
        )
        assert r.status_code == 200
        assert r.json()["results"][0]["ok"] is False


async def test_upload_validation(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "v"})).json()

        # unsupported extension
        r = await http.post(
            f"/knowledge/{kb['id']}/files",
            files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
        )
        body = r.json()
        assert r.status_code == 200 and body["results"][0]["ok"] is False
        assert "Unsupported file type" in body["results"][0]["error"]

        # path traversal rejected
        r = await http.post(
            f"/knowledge/{kb['id']}/files",
            files=[("files", ("x.md", b"# hi", "text/markdown"))],
            data={"paths": "../../etc/passwd"},
        )
        assert r.json()["results"][0]["ok"] is False

        # oversized file rejected
        old = config.settings.kb_max_file_size_mb
        config.settings.kb_max_file_size_mb = 0
        try:
            r = await http.post(
                f"/knowledge/{kb['id']}/files",
                files=[("files", ("big.md", b"x" * 100, "text/markdown"))],
            )
        finally:
            config.settings.kb_max_file_size_mb = old
        body = r.json()
        assert r.status_code == 200 and body["results"][0]["ok"] is False
        assert "too large" in body["results"][0]["error"]

        # unparseable content -> document stored with status failed
        r = await http.post(
            f"/knowledge/{kb['id']}/files",
            files=[("files", ("broken.md", b"   \n \n  ", "text/markdown"))],
        )
        body = r.json()
        assert r.status_code == 200 and body["results"][0]["ok"] is False
        assert "No extractable text" in body["results"][0]["error"]
        docs = (await http.get(f"/knowledge/{kb['id']}/files")).json()
        failed = next(d for d in docs if d["path"] == "broken.md")
        assert failed["status"] == "failed" and failed["error"]


async def test_delete_cleans_vectors_and_reindex(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "kb"})).json()
        kb_id = kb["id"]
        r = await http.post(
            f"/knowledge/{kb_id}/files",
            files=[
                (
                    "files",
                    ("a.md", b"# Alpha\n\nAlpha content about alpha things.", "text/markdown"),
                )
            ],
        )
        doc_id = r.json()["results"][0]["doc_id"]

        r = await http.get(f"/knowledge/{kb_id}/search", params={"q": "alpha"})
        assert r.json()["hits"], "expected a hit before deletion"

        # delete the document -> vectors gone
        r = await http.delete(f"/knowledge/{kb_id}/files/{doc_id}")
        assert r.status_code == 204
        r = await http.get(f"/knowledge/{kb_id}/search", params={"q": "alpha"})
        assert r.json()["hits"] == []

        # reindex a second document
        r = await http.post(
            f"/knowledge/{kb_id}/files",
            files=[
                ("files", ("b.md", b"# Beta\n\nBeta content about beta things.", "text/markdown"))
            ],
        )
        r = await http.post(f"/knowledge/{kb_id}/reindex")
        assert r.status_code == 200 and r.json()["processed"] == 1, r.text
        r = await http.get(f"/knowledge/{kb_id}/search", params={"q": "beta"})
        assert r.json()["hits"], "expected a hit after reindex"

        # deleting the KB also cleans the vector store
        r = await http.delete(f"/knowledge/{kb_id}")
        assert r.status_code == 204
        assert kb_env._chunks == []


async def test_search_requires_vector_store(kb_env):
    """Without a configured vector store the search endpoint returns 503."""
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "x"})).json()
        reset_vector_store()  # simulate WEAVIATE_URL unset
        try:
            r = await http.get(f"/knowledge/{kb['id']}/search", params={"q": "anything"})
        finally:
            set_vector_store(kb_env)
        assert r.status_code == 503, r.text


# ---------------------------------------------------------------------------
# hardening: zip upload, download, quota, global search
# ---------------------------------------------------------------------------


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries:
            zf.writestr(name, content)
    return buf.getvalue()


async def test_zip_upload(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "zipped"})).json()
        archive = _zip_bytes(
            [
                ("docs/intro.md", b"# Intro\n\nIntro about the zipped docs."),
                ("docs/notes.txt", b"Zipped notes about packaging."),
            ]
        )
        r = await http.post(
            f"/knowledge/{kb['id']}/zip",
            files=[("file", ("docs.zip", archive, "application/zip"))],
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert all(res["ok"] for res in body["results"]), body
        assert {res["path"] for res in body["results"]} == {
            "docs/intro.md",
            "docs/notes.txt",
        }

        docs = (await http.get(f"/knowledge/{kb['id']}/files")).json()
        assert {d["path"] for d in docs} == {"docs/intro.md", "docs/notes.txt"}
        assert all(d["status"] == "ready" for d in docs)
        r = await http.get(f"/knowledge/{kb['id']}/search", params={"q": "packaging"})
        assert r.json()["hits"][0]["path"] == "docs/notes.txt"


async def test_zip_upload_guards(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "guarded"})).json()

        # path traversal entry aborts the whole archive
        r = await http.post(
            f"/knowledge/{kb['id']}/zip",
            files=[("file", ("evil.zip", _zip_bytes([("../evil.md", b"# x")]), "application/zip"))],
        )
        body = r.json()
        assert r.status_code == 200 and body["results"][0]["ok"] is False
        assert "Unsafe path" in body["results"][0]["error"]

        # not a zip
        r = await http.post(
            f"/knowledge/{kb['id']}/zip",
            files=[("file", ("fake.zip", b"not a zip at all", "application/zip"))],
        )
        assert "Not a valid zip" in r.json()["results"][0]["error"]

        # total size guard (0 MB cap rejects any content)
        old = config.settings.kb_zip_max_total_mb
        config.settings.kb_zip_max_total_mb = 0
        try:
            r = await http.post(
                f"/knowledge/{kb['id']}/zip",
                files=[("file", ("big.zip", _zip_bytes([("a.md", b"# hi")]), "application/zip"))],
            )
        finally:
            config.settings.kb_zip_max_total_mb = old
        assert "total size" in r.json()["results"][0]["error"]

        # per-entry extension check (soft: other entries still processed)
        r = await http.post(
            f"/knowledge/{kb['id']}/zip",
            files=[
                (
                    "file",
                    (
                        "mixed.zip",
                        _zip_bytes([("ok.md", b"# ok"), ("bad.exe", b"MZ")]),
                        "application/zip",
                    ),
                )
            ],
        )
        results = {res["path"]: res for res in r.json()["results"]}
        assert results["ok.md"]["ok"] is True
        assert results["bad.exe"]["ok"] is False
        assert "Unsupported file type" in results["bad.exe"]["error"]

        # nothing was stored from the rejected archives
        docs = (await http.get(f"/knowledge/{kb['id']}/files")).json()
        assert {d["path"] for d in docs} == {"ok.md"}


async def test_download_document_content(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "dl"})).json()
        payload = b"# Hello\n\nDownload me."
        r = await http.post(
            f"/knowledge/{kb['id']}/files",
            files=[("files", ("hello.md", payload, "text/markdown"))],
        )
        doc_id = r.json()["results"][0]["doc_id"]

        r = await http.get(f"/knowledge/{kb['id']}/files/{doc_id}/content")
        assert r.status_code == 200
        assert r.content == payload
        assert r.headers["content-type"].startswith("text/markdown")
        assert "inline" in r.headers["content-disposition"]
        assert "hello.md" in r.headers["content-disposition"]

        # owner isolation
        async with await _client_for(app, username="other") as other:
            r = await other.get(f"/knowledge/{kb['id']}/files/{doc_id}/content")
            assert r.status_code == 404


async def test_quota_enforced(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb = (await http.post("/knowledge", json={"name": "quota"})).json()
        old = config.settings.kb_quota_mb
        config.settings.kb_quota_mb = 0  # no free space at all
        try:
            r = await http.post(
                f"/knowledge/{kb['id']}/files",
                files=[("files", ("a.md", b"# hi", "text/markdown"))],
            )
            r2 = await http.post(
                f"/knowledge/{kb['id']}/zip",
                files=[("file", ("a.zip", _zip_bytes([("b.md", b"# yo")]), "application/zip"))],
            )
        finally:
            config.settings.kb_quota_mb = old
        assert r.json()["results"][0]["error"] == "Storage quota exceeded"
        assert r2.json()["results"][0]["error"] == "Storage quota exceeded"
        assert (await http.get(f"/knowledge/{kb['id']}/files")).json() == []


async def test_global_search_across_kbs(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=_scripted_model(),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _client_for(app) as http:
        kb1 = (await http.post("/knowledge", json={"name": "one"})).json()
        kb2 = (await http.post("/knowledge", json={"name": "two"})).json()
        await http.post(
            f"/knowledge/{kb1['id']}/files",
            files=[("files", ("a.md", b"# Alpha\n\nAlpha documentation.", "text/markdown"))],
        )
        await http.post(
            f"/knowledge/{kb2['id']}/files",
            files=[("files", ("b.md", b"# Beta\n\nBeta documentation.", "text/markdown"))],
        )
        r = await http.get("/knowledge/search", params={"q": "documentation", "limit": 10})
        assert r.status_code == 200, r.text
        paths = {hit["path"] for hit in r.json()["hits"]}
        assert paths == {"a.md", "b.md"}, paths


# ---------------------------------------------------------------------------
# agent tool (full SSE pipeline, context.user_id resolution)
# ---------------------------------------------------------------------------


async def test_agent_kb_tool_scoped_to_user(kb_env, memory_persistence):
    """The search_knowledge_base tool sees only the runtime user's chunks."""
    # Seed a KB + document for "tester" via the service layer (same stores the
    # agent's tool reads through get_vector_store()).
    kb = await persistence.kb.create_kb("tester", "docs", None)
    doc = await persistence.kb.add_document(
        "tester", kb["id"], "notes/deploy.md", "text/markdown", 42, b"x"
    )
    from app.services.kb.ingest import ingest_document

    await ingest_document(doc, b"# Deployment\n\nDeploy with kubectl rollout.", kb_env)

    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="search_knowledge_base",
                        args={"query": "kubectl deployment", "top_k": 3},
                    )
                ],
            ),
            AIMessage(content="Deployment uses kubectl rollout."),
        ]
    )

    def agent_for() -> Any:
        return build_agent(
            checkpointer=memory_persistence.checkpointer,
            store=memory_persistence.store,
            extra_tools=[build_kb_search_tool(vector_store=kb_env)],
            model=Scripted(responses=list(model.responses)),
            system_prompt="test",
        )

    # tester sees the passage
    events = await collect_stream(agent_for(), "tester", message="how do we deploy?")
    tool_end = next(
        d for e, d in events if e == "tool_end" and d["name"] == "search_knowledge_base"
    )
    output = tool_end["output"]["content"]
    assert "notes/deploy.md" in output, output
    assert "kubectl" in output

    # another user gets no results (isolation via runtime context)
    events = await collect_stream(agent_for(), "other", message="how do we deploy?")
    tool_end = next(
        d for e, d in events if e == "tool_end" and d["name"] == "search_knowledge_base"
    )
    output = tool_end["output"]["content"]
    assert "No matching passages" in output, output


# ---------------------------------------------------------------------------
# page-level chunking
# ---------------------------------------------------------------------------


async def test_chunk_document_page_level(kb_env):
    # pages within the budget stay whole (PDF page-level behavior)
    chunks = chunk_document(
        "guide.pdf", ["short page one", "short page two"], chunk_size=100, chunk_overlap=20
    )
    assert chunks == ["short page one", "short page two"]

    # oversized pages fall back to the recursive splitter
    long_page = "word " * 300
    chunks = chunk_document("guide.pdf", [long_page], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 + 20 for c in chunks)

    # empty pages are skipped
    chunks = chunk_document(
        "guide.pdf", ["", "   ", "real content"], chunk_size=100, chunk_overlap=20
    )
    assert chunks == ["real content"]

    # markdown stays header-aware with header path prefixes
    md = textwrap.dedent(
        """\
        # Deployment
        How we deploy.

        ## Rollback
        How we roll back.
        """
    )
    chunks = chunk_document("runbook.md", [md], chunk_size=1000, chunk_overlap=100)
    assert any("Deployment" in c and "How we deploy" in c for c in chunks)
    assert any("Rollback" in c and "How we roll back" in c for c in chunks)


async def test_extract_pages_non_pdf(kb_env):
    # non-PDF formats produce a single page
    pages = extract_pages("notes.txt", b"hello world")
    assert pages == ["hello world"]


# ---------------------------------------------------------------------------
# per-request alpha on search endpoints
# ---------------------------------------------------------------------------


async def _api_client(app, username: str = "tester") -> httpx.AsyncClient:
    await persistence.users.create_user(username=username, hashed_password="x")
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_search_alpha_param(kb_env):
    app = create_app(
        agent=build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            model=Scripted(responses=[AIMessage(content="ok")]),
            system_prompt="test",
        )
    )
    async with app.router.lifespan_context(app), await _api_client(app) as http:
        kb = (await http.post("/knowledge", json={"name": "alpha-test"})).json()
        await http.post(
            f"/knowledge/{kb['id']}/files",
            files=[
                (
                    "files",
                    ("deploy.md", b"# Deployment\n\nkubectl deployment rollout.", "text/markdown"),
                )
            ],
        )
        # alpha=0 -> keyword-only; alpha=1 -> vector-only; both must still work
        for alpha in (0.0, 1.0):
            r = await http.get(
                f"/knowledge/{kb['id']}/search", params={"q": "deployment", "alpha": alpha}
            )
            assert r.status_code == 200, r.text
            assert r.json()["hits"], f"expected hits at alpha={alpha}"
        # out-of-range alpha rejected
        r = await http.get(
            f"/knowledge/{kb['id']}/search", params={"q": "deployment", "alpha": 1.5}
        )
        assert r.status_code == 422
        # global search also accepts alpha
        r = await http.get("/knowledge/search", params={"q": "deployment", "alpha": 0.5})
        assert r.status_code == 200 and r.json()["hits"]


async def test_bm25_property_weights_config(kb_env):
    """KB_BM25_PROPERTY_WEIGHTS parsing: empty dict disables, default boosts path."""
    assert isinstance(config.settings.kb_bm25_property_weights, dict)
    from app.services.kb.vectorstore import WeaviateKbVectorStore

    # only reachable statically (no Weaviate connection in offline tests)
    assert WeaviateKbVectorStore._bm25_properties() == ["path^2.0"]
    old = config.settings.kb_bm25_property_weights
    config.settings.kb_bm25_property_weights = {}
    try:
        assert WeaviateKbVectorStore._bm25_properties() is None
    finally:
        config.settings.kb_bm25_property_weights = old
