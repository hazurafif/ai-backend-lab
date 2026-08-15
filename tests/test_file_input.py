"""Offline tests for chat file uploads (multipart /chat + /api/chat).

Files are saved under <UPLOADS_DIR>/<username>/ with sanitized names and
size caps; the agent is told their paths in the user message so it can
manipulate them with its filesystem/execute tools. No API key / network /
Postgres.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services import uploads
from app.services.agent import build_agent

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


class Scripted(BaseChatModel):
    """Scripted model that records the message lists it is invoked with."""

    responses: list[AIMessage] = Field(default_factory=lambda: [AIMessage(content="ok")])
    seen: list[list[Any]] = Field(default_factory=list)
    tools: Sequence[dict | type] = ()

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
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses[0])])

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
async def memory_persistence(tmp_path, monkeypatch):
    """In-memory backend + workspace root redirected to a temp dir."""
    config.settings.database_uri = None
    monkeypatch.setattr(config.settings, "workspace_root", str(tmp_path / "workspace"))
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


async def _client(app, username: str) -> httpx.AsyncClient:
    await database.persistence.users.create_user(
        username=username, hashed_password="x", role="admin"
    )
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def _agent_app(model: Scripted) -> tuple[object, Scripted]:
    app = create_app(
        agent=build_agent(
            checkpointer=database.persistence.checkpointer,
            store=database.persistence.store,
            mcp_tools=[],
            model=model,
            system_prompt="test",
        )
    )
    return app, model


# ---------------------------------------------------------------------------
# sanitize / save helpers
# ---------------------------------------------------------------------------


def test_sanitize_filename():
    assert uploads.sanitize_filename("../evil.pdf") == "evil.pdf"
    assert uploads.sanitize_filename("a/b/c.txt") == "c.txt"
    assert uploads.sanitize_filename("report v2.pdf") == "report v2.pdf"
    assert uploads.sanitize_filename("") == "upload"
    assert uploads.sanitize_filename("..") == "upload"
    assert uploads.sanitize_filename("../../etc/passwd") == "passwd"


async def test_save_upload_writes_file_and_dedupes(memory_persistence):
    from app.services.workspace import workspace_dir

    class FakeUpload:
        filename = "report.pdf"
        content_type = "application/pdf"
        _done = False

        async def read(self, n: int) -> bytes:
            if self._done:
                return b""
            self._done = True
            return b"%PDF-1.4 fake"

        async def close(self) -> None:
            pass

    r1 = await uploads.save_upload("alice", FakeUpload())
    assert r1["name"] == "report.pdf"
    assert r1["size"] == len(b"%PDF-1.4 fake")
    assert r1["path"] == "/uploads/report.pdf"
    # A real file in alice's workspace (versioned by its git repo).
    uploaded = workspace_dir("alice") / "uploads" / "report.pdf"
    assert uploaded.read_bytes() == b"%PDF-1.4 fake"

    # Same name again -> numeric suffix, both kept.
    r2 = await uploads.save_upload("alice", FakeUpload())
    assert r2["name"] == "report (1).pdf"
    assert r2["path"] == "/uploads/report (1).pdf"
    assert (workspace_dir("alice") / "uploads" / "report (1).pdf").exists()


async def test_upload_lands_in_workspace(memory_persistence):
    """Uploads are real files in the user's workspace, readable by file tools."""
    from app.services.workspace import workspace_dir

    class FakeUpload:
        filename = "data.csv"
        content_type = "text/csv"
        _done = False

        async def read(self, n: int) -> bytes:
            if self._done:
                return b""
            self._done = True
            return b"a,b\n1,2\n"

        async def close(self) -> None:
            pass

    r = await uploads.save_upload("alice", FakeUpload())
    assert "error" not in r

    # The file tool path /uploads/<name> resolves into alice's workspace.
    target = workspace_dir("alice") / "uploads" / "data.csv"
    assert target.read_text() == "a,b\n1,2\n"

    # Outside a graph run the backend user is anonymous; the sync-free model
    # keeps the file purely on disk (no store mirror).
    assert await memory_persistence.store.aget(("alice",), "/alice/data.csv") is None

    # Oversized uploads never reach the store.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config.settings, "max_upload_size_mb", 0)

    class BigUpload:
        filename = "big.bin"
        content_type = "application/octet-stream"

        async def read(self, n: int) -> bytes:
            return b"x" * n

        async def close(self) -> None:
            pass

    try:
        r = await uploads.save_upload("bob", BigUpload())
    finally:
        monkeypatch.undo()
    assert "error" in r
    assert not (workspace_dir("bob") / "uploads" / "big.bin").exists()


async def test_upload_over_cap_is_skipped_with_note(memory_persistence, monkeypatch):
    monkeypatch.setattr(config.settings, "max_upload_size_mb", 0)  # anything > 0 bytes fails

    class BigUpload:
        filename = "big.bin"
        content_type = "application/octet-stream"

        async def read(self, n: int) -> bytes:
            return b"x" * n

        async def close(self) -> None:
            pass

    result = await uploads.save_upload("alice", BigUpload())
    assert "error" in result and "skipped" in result["error"]
    item = await memory_persistence.store.aget(("alice",), "/alice/big.bin")
    assert item is None


# ---------------------------------------------------------------------------
# HTTP: /chat multipart
# ---------------------------------------------------------------------------


async def test_chat_multipart_upload_streams_and_mentions_path(memory_persistence):
    model = Scripted()
    app, model = _agent_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        r = await client.post(
            "/chat",
            data={"message": "summarize this pdf"},
            files={"files": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert r.status_code == 200, r.text
        assert "event: message_delta" in r.text
        assert "event: done" in r.text

    # The agent's first (human) message mentions the uploaded file path.
    assert model.seen, "scripted model was never invoked"
    last_human = next(m for m in reversed(model.seen[-1]) if getattr(m, "type", "") == "human")
    text = str(last_human.content)
    assert "report.pdf" in text
    assert "uploads/" in text
    assert "pdftotext" in text  # manipulation hint (execute)


async def test_chat_multipart_rejects_empty_message(memory_persistence):
    model = Scripted()
    app, _model = _agent_app(model)
    async with app.router.lifespan_context(app), await _client(app, "alice") as client:
        r = await client.post("/chat", data={"message": "   "})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# HTTP: /api/chat multipart (AI SDK useChat attachments)
# ---------------------------------------------------------------------------


async def test_api_chat_multipart_upload(memory_persistence):
    model = Scripted()
    app, model = _agent_app(model)
    messages = [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "read the csv"}]}]
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        r = await client.post(
            "/api/chat",
            data={"id": "chat-upload", "messages": json.dumps(messages)},
            files={"files": ("data.csv", b"a,b,c\n1,2,3\n", "text/csv")},
        )
        assert r.status_code == 200, r.text
        assert '"type": "text-delta"' in r.text
        assert "data: [DONE]" in r.text

    assert model.seen
    last_human = next(m for m in reversed(model.seen[-1]) if getattr(m, "type", "") == "human")
    text = str(last_human.content)
    assert "data.csv" in text and "uploads/" in text
