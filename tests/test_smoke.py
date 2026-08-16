"""Offline end-to-end tests — no API key, no network, no Postgres.

Uses a scripted fake chat model (same trick as deepagents' own tests) that
returns a fixed sequence of responses. Exercises the full pipeline:

    HTTP POST /chat  ->  agent astream_events(v3)  ->  SSE events
    interrupt + resume (human-in-the-loop)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from app.core.config import settings
from app.core.database import persistence
from app.core.security import create_access_token
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


# ---------------------------------------------------------------------------
# fixtures & helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memory_persistence():
    """Force in-memory checkpointer/store and start the app singleton."""
    settings.database_uri = None
    await persistence.start()
    yield persistence
    await persistence.stop()


@tool
def echo(x: str) -> str:
    """Echo the input back with a marker."""
    return f"echo:{x}"


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


def build_scripted_agent(checkpointer, store, *, interrupt_on: dict | None = None):
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-1", name="echo", args={"x": "hello"})],
            ),
            AIMessage(content="Final answer from the agent."),
        ]
    )
    return build_agent(
        checkpointer=checkpointer,
        store=store,
        mcp_tools=[echo],
        model=model,
        system_prompt="test",
        interrupt_on=interrupt_on,
    )


def build_reasoning_agent(checkpointer, store, *, reasoning: str, answer: str):
    """Agent whose model emits a reasoning block followed by the answer text."""
    model = Scripted(
        responses=[
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": reasoning},
                    {"type": "text", "text": answer},
                ]
            )
        ]
    )
    return build_agent(
        checkpointer=checkpointer,
        store=store,
        mcp_tools=[],
        model=model,
        system_prompt="test",
    )


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


def parse_sdk_data_lines(text: str) -> list[dict]:
    """Parse the `data:` lines of an AI SDK data-stream response."""
    chunks: list[dict] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line.removeprefix("data: ")
            if payload != "[DONE]":
                chunks.append(json.loads(payload))
    return chunks


async def collect_stream(agent, username, **kwargs) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, username, **kwargs):
        events.append(parse_sse_chunk(chunk))
    return events


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


async def test_direct_streaming_pipeline(memory_persistence):
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
    events = await collect_stream(agent, "tester", message="hi there")

    names = [e for e, _ in events]
    assert "message_delta" in names, "missing message_delta"
    assert "message" in names, "missing finalized message"
    assert "tool_start" in names, "missing tool_start"
    assert "tool_end" in names, "missing tool_end"
    done = [d for e, d in events if e == "done"]
    assert done and done[-1].get("messages"), "missing done with messages"
    assert done[-1].get("interrupted") is None, "should not be interrupted"

    # Chat history rows match the finalized messages of the done event.
    rows = await memory_persistence.chat_history.list_messages(done[-1]["thread_id"])
    assert [r["type"] for r in rows] == [m["type"] for m in done[-1]["messages"]]

    tool_end = next(d for e, d in events if e == "tool_end")
    assert tool_end["name"] == "echo"
    assert tool_end["output"]["content"] == "echo:hello", tool_end


async def test_reasoning_streaming_pipeline(memory_persistence):
    """Thinking content is streamed live as reasoning_delta events."""
    agent = build_reasoning_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        reasoning="Let me reason step by step: the answer is 42.",
        answer="The answer is 42.",
    )
    events = await collect_stream(agent, "tester", message="think hard")

    names = [e for e, _ in events]
    assert "reasoning_start" in names, "missing reasoning_start"
    assert "reasoning_delta" in names, "missing reasoning_delta events"
    assert "reasoning_end" in names, "missing reasoning_end"
    assert "message_delta" in names, "missing message_delta events"

    # Thinking is a bracketed lifecycle (start -> delta* -> end), shaped
    # like a tool call, with a stable id across the whole turn.
    reasoning = [d for e, d in events if e == "reasoning_delta"]
    assert "".join(d["delta"] for d in reasoning) == "Let me reason step by step: the answer is 42."
    first_idx = names.index("reasoning_start")
    deltas = [i for i, e in enumerate(names) if e == "reasoning_delta"]
    end_idx = names.index("reasoning_end")
    assert deltas and first_idx < deltas[0] and deltas[-1] < end_idx
    starts = [d for e, d in events if e == "reasoning_start"]
    ends = [d for e, d in events if e == "reasoning_end"]
    assert len(starts) == len(ends) == 1
    assert starts[0]["id"] == ends[0]["id"] == reasoning[0]["id"]
    text = "".join(d["delta"] for e, d in events if e == "message_delta")
    assert text == "The answer is 42."
    msg_idx = names.index("message")
    assert end_idx < msg_idx

    # The finalized message keeps the reasoning block (langchain schema).
    msg = next(d for e, d in events if e == "message")
    content = msg["message"]["content"]
    assert content[0]["type"] == "reasoning"
    assert content[0]["reasoning"] == "Let me reason step by step: the answer is 42."
    assert content[1]["type"] == "text"

    # The stored thread history preserves the reasoning block too.
    done = [d for e, d in events if e == "done"]
    stored = await memory_persistence.chat_history.list_messages(done[-1]["thread_id"])
    assert stored and stored[-1]["content"][0]["type"] == "reasoning"


async def test_hide_tool_calls_preference(memory_persistence):
    """`hide_tool_calls` hides the tool card, not the data.

    The per-user display preference (PATCH /users/me/preferences) marks the
    tool lifecycle events `"hidden": true` instead of dropping them: the
    tool output (e.g. web_search sources) keeps streaming so citation links
    stay clickable, and finalized `message` / `done.messages` payloads keep
    tool-call fields and tool rows for the same reason. The persisted chat
    history stays complete — the durable record is never scrubbed.
    """
    await memory_persistence.preferences.set("tester", "hide_tool_calls", True)
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
    events = await collect_stream(agent, "tester", message="hi there")

    # The tool lifecycle still streams, marked hidden.
    starts = [d for e, d in events if e == "tool_start"]
    assert starts, "tool_start must still stream (data needed for citations)"
    assert all(d.get("hidden") is True for d in starts), starts
    ends = [d for e, d in events if e == "tool_end"]
    assert ends and all(d.get("hidden") is True for d in ends), ends
    assert all(d.get("hidden") is True for e, d in events if e == "tool_delta")
    assert "message_delta" in [e for e, _ in events], "answer text still streams"

    # Finalized message events KEEP tool-call fields (citations resolvable).
    messages = [d["message"] for e, d in events if e == "message"]
    assert messages, "missing message events"
    assert any("tool_calls" in m or "tool_call_chunks" in m for m in messages)

    # done.messages keeps tool-result rows and tool-call fields.
    done = [d for e, d in events if e == "done"][-1]
    assert [m for m in done["messages"] if m.get("type") == "tool"], done["messages"]
    ai_rows = [m for m in done["messages"] if m.get("type") == "ai"]
    assert ai_rows and any("tool_calls" in m for m in ai_rows), ai_rows

    # The durable record keeps the full conversation (tool rows included).
    rows = await memory_persistence.chat_history.list_messages(done["thread_id"])
    assert any(m.get("type") == "tool" for m in rows), "history must keep tool rows"


async def test_hide_reasoning_preference(memory_persistence):
    """`hide_reasoning` drops the thinking bracket and scrubs reasoning blocks."""
    await memory_persistence.preferences.set("tester", "hide_reasoning", True)
    agent = build_reasoning_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        reasoning="Let me reason step by step: the answer is 42.",
        answer="The answer is 42.",
    )
    events = await collect_stream(agent, "tester", message="think hard")

    names = [e for e, _ in events]
    assert not any(n.startswith("reasoning_") for n in names), names
    text = "".join(d["delta"] for e, d in events if e == "message_delta")
    assert text == "The answer is 42.", text

    # Finalized message content keeps only the text block.
    msg = next(d for e, d in events if e == "message")
    content = msg["message"]["content"]
    assert content and not [b for b in content if b.get("type") == "reasoning"], content

    done = [d for e, d in events if e == "done"][-1]
    ai = next(m for m in done["messages"] if m.get("type") == "ai")
    assert not [b for b in ai["content"] if b.get("type") == "reasoning"], ai["content"]


async def test_hide_preferences_are_per_user(memory_persistence):
    """Another user without the preferences still sees the full stream."""
    await memory_persistence.preferences.set("tester", "hide_tool_calls", True)
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)

    hidden = await collect_stream(agent, "tester", message="hi there")
    starts = [d for e, d in hidden if e == "tool_start"]
    assert starts and all(d.get("hidden") is True for d in starts), starts

    # Fresh agent: the scripted model is a one-shot sequence per instance.
    fresh = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
    visible = await collect_stream(fresh, "someone-else", message="hi there")
    starts = [d for e, d in visible if e == "tool_start"]
    assert starts and all("hidden" not in d for d in starts), starts


async def test_interrupt_and_resume(memory_persistence):
    agent = build_scripted_agent(
        memory_persistence.checkpointer,
        memory_persistence.store,
        interrupt_on={"echo": True},
    )

    # Run 1: the tool call is paused for human approval.
    events = await collect_stream(agent, "tester", message="approve this tool call")
    names = [e for e, _ in events]
    assert "interrupt" in names, "missing interrupt event"

    interrupt = next(d for e, d in events if e == "interrupt")
    action_requests = interrupt["interrupts"][0]["action_requests"]
    assert action_requests[0]["name"] == "echo", action_requests
    assert action_requests[0]["args"] == {"x": "hello"}

    done = [d for e, d in events if e == "done"][-1]
    assert done["interrupted"] is True
    assert not any(e == "tool_start" for e in names), "tool must NOT run before approval"
    thread_id = done["thread_id"]

    # Resume with approval -> tool executes, final answer streams.
    events2 = await collect_stream(
        agent, "tester", thread_id=thread_id, resume={"decisions": [{"type": "approve"}]}
    )
    names2 = [e for e, _ in events2]
    assert "tool_start" in names2 and "tool_end" in names2, "tool should run after approval"
    done2 = [d for e, d in events2 if e == "done"][-1]
    assert not done2.get("interrupted"), "run should complete after resume"

    def extract_text(m: dict) -> str:
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(b.get("text", "") for b in c if b.get("type") == "text")
        return ""

    texts = [extract_text(m) for m in done2["messages"] if m.get("type") == "ai"]
    assert any("Final answer" in t for t in texts), texts

    # Chat history: rows written at interrupt (human + tool-call AI) and the
    # resumed run appended the rest — no duplicates across the two runs.
    rows = await memory_persistence.chat_history.list_messages(thread_id)
    assert [r["type"] for r in rows] == [m["type"] for m in done2["messages"]]
    assert len(rows) == len(done2["messages"]) == 4, rows
    assert len({r.get("id") for r in rows}) == len(rows), "duplicate message ids"


async def test_ai_sdk_extract_user_message():
    from app.services.ai_sdk_chat import extract_user_message

    # parts format (what useChat sends)
    parts = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hello via sdk"}]},
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hi"}]},
        {"id": "u2", "role": "user", "parts": [{"type": "text", "text": "second msg"}]},
    ]
    assert extract_user_message(parts) == "second msg"
    # legacy string content
    assert extract_user_message([{"role": "user", "content": "plain"}]) == "plain"
    # empty / no user message
    assert extract_user_message([]) == ""
    assert extract_user_message([{"role": "assistant", "content": "x"}]) == ""


async def test_ai_sdk_chat_endpoint(memory_persistence):
    """POST /api/chat speaks the AI SDK data-stream protocol (useChat)."""
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        _agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
        app.state.agent = _agent
        app.state.agents.set_static_default(_agent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "id": "chat-abc",
                    "selectedChatModel": "gpt-4o-mini",
                    "messages": [
                        {
                            "id": "m1",
                            "role": "user",
                            "parts": [{"type": "text", "text": "hello via sdk"}],
                        }
                    ],
                },
            )
            assert r.status_code == 200, r.text

            chunks = parse_sdk_data_lines(r.text)
            types = [c["type"] for c in chunks]
            assert types[0] == "start", types
            assert "text-start" in types, "missing text-start"
            assert "text-delta" in types, "missing text-delta"
            assert "text-end" in types, "missing text-end"
            assert types[-1] == "finish", types
            assert chunks[-1]["finishReason"] == "stop"
            assert "data: [DONE]" in r.text

            # tool activity surfaces as native AI SDK tool chunks (echo tool in scripted agent)
            tool_input = [c for c in chunks if c["type"] == "tool-input-start"]
            assert tool_input, [c["type"] for c in chunks]
            assert tool_input[0]["toolName"] == "echo"
            assert any(c["type"] == "tool-input-available" for c in chunks)
            assert any(c["type"] == "tool-output-available" for c in chunks)

            tool_available = next(c for c in chunks if c["type"] == "tool-input-available")
            assert tool_available["input"] == {"x": "hello"}, tool_available

            # text of the answer is streamed verbatim
            text = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
            assert "Final answer from the agent." in text, text

            # no user message -> 422
            r = await client.post("/api/chat", json={"id": "x", "messages": []})
            assert r.status_code == 422, r.text


async def test_ai_sdk_hide_tool_calls_keeps_tool_output(memory_persistence):
    """AI SDK path (POST /api/chat): hide_tool_calls hides the card, not the data.

    The tool lifecycle still translates to native tool-input-*/tool-output-*
    chunks so the web_search output (citation sources) reaches the frontend
    even when the user hides tool calls.
    """
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        # Persistence re-initializes on lifespan entry: set prefs + user after.
        await memory_persistence.preferences.set("tester", "hide_tool_calls", True)
        await memory_persistence.users.create_user(username="tester", hashed_password="x")
        token = create_access_token(data={"sub": "tester"})
        _agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
        app.state.agent = _agent
        app.state.agents.set_static_default(_agent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            r = await client.post(
                "/api/chat",
                json={"id": "chat-hide-tools", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text
            chunks = parse_sdk_data_lines(r.text)
            types = [c["type"] for c in chunks]
            # Tool DATA keeps flowing (citations stay clickable)...
            assert "tool-input-start" in types, types
            outputs = [c for c in chunks if c["type"] == "tool-output-available"]
            assert outputs, "tool output must still stream under hide_tool_calls"
            # ...and no reasoning is involved here.
            assert not any(t.startswith("reasoning") for t in types), types
            assert "text-delta" in types, types


async def test_ai_sdk_hide_reasoning_drops_thinking(memory_persistence):
    """AI SDK path (POST /api/chat): hide_reasoning drops the thinking bracket.

    The reasoning_start/delta/end events never reach the bridge, so no
    reasoning-* chunks are emitted on the data stream.
    """
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        # Persistence re-initializes on lifespan entry: set prefs + user after.
        await memory_persistence.preferences.set("tester", "hide_reasoning", True)
        await memory_persistence.users.create_user(username="tester", hashed_password="x")
        token = create_access_token(data={"sub": "tester"})
        _agent = build_reasoning_agent(
            memory_persistence.checkpointer,
            memory_persistence.store,
            reasoning="Let me think step by step",
            answer="The answer is 42.",
        )
        app.state.agent = _agent
        app.state.agents.set_static_default(_agent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            r = await client.post(
                "/api/chat",
                json={"id": "chat-hide-reason", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text
            chunks = parse_sdk_data_lines(r.text)
            types = [c["type"] for c in chunks]
            assert not any(t.startswith("reasoning") for t in types), types
            # The answer still streams.
            text = "".join(c.get("delta", "") for c in chunks if c["type"] == "text-delta")
            assert text == "The answer is 42.", text


async def test_chat_history_persisted_without_duplicates(memory_persistence):
    """Every turn appends exactly its messages to the chat history table."""
    agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)

    done1 = [
        d for e, d in await collect_stream(agent, "tester", message="turn one") if e == "done"
    ][-1]
    thread_id = done1["thread_id"]
    rows = await memory_persistence.chat_history.list_messages(thread_id)
    assert [r["type"] for r in rows] == [m["type"] for m in done1["messages"]]

    # Second turn on the same thread: new messages appended, old ones kept.
    done2 = [
        d
        for e, d in await collect_stream(agent, "tester", thread_id=thread_id, message="turn two")
        if e == "done"
    ][-1]
    rows = await memory_persistence.chat_history.list_messages(thread_id)
    assert [r["type"] for r in rows] == [m["type"] for m in done2["messages"]]
    assert len(rows) == len(done2["messages"])
    assert len({r.get("id") for r in rows}) == len(rows), "duplicate message ids"

    # Thread metadata: title from the first message, updated_at refreshed.
    item = await memory_persistence.store.aget(("threads", "tester"), thread_id)
    assert item.value["title"] == "turn one"
    assert item.value["updated_at"] >= item.value["created_at"]


async def test_http_end_to_end(memory_persistence):
    # Placeholder agent so lifespan skips the real (OpenAI) model; the
    # scripted agent is built on the SAME persistence instances the API uses.
    app = create_app(agent=build_scripted_agent(InMemorySaver(), InMemoryStore()))

    async with app.router.lifespan_context(app):
        _agent = build_scripted_agent(memory_persistence.checkpointer, memory_persistence.store)
        app.state.agent = _agent
        app.state.agents.set_static_default(_agent)
        # get_current_user validates against the users store.
        await memory_persistence.users.create_user(username="tester", hashed_password="x")
        token = create_access_token(data={"sub": "tester"})

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # auth required
            r = await client.get("/threads")
            assert r.status_code == 401

            headers = {"Authorization": f"Bearer {token}"}
            r = await client.post("/chat", json={"message": "hello via http"}, headers=headers)
            assert r.status_code == 200, r.text
            assert "event: message_delta" in r.text
            assert "event: done" in r.text

            r = await client.get("/threads", headers=headers)
            assert r.status_code == 200
            threads = r.json()
            assert len(threads) >= 1 and threads[0]["title"] == "hello via http", threads

            thread_id = threads[0]["thread_id"]
            r = await client.get(f"/threads/{thread_id}/messages", headers=headers)
            assert r.status_code == 200 and len(r.json()) >= 2, r.text

            # Legacy threads (no rows in the chat_messages table) fall back to
            # checkpoint rehydration — same response shape.
            memory_persistence.chat_history._memory.clear()
            memory_persistence.chat_history._memory_ids.clear()
            r = await client.get(f"/threads/{thread_id}/messages", headers=headers)
            assert r.status_code == 200 and len(r.json()) >= 2, r.text

            # resume on a non-interrupted thread -> 409
            r = await client.post(
                f"/threads/{thread_id}/resume",
                json={"decision": {"type": "approve"}},
                headers=headers,
            )
            assert r.status_code == 409, r.text


async def test_agent_publishes_skill_to_store(memory_persistence):
    """The publish_skill tool persists a drafted skill into the user's store namespace.

    The agent drafts a skill folder in its workspace (SKILL.md + bundled
    files), then calls publish_skill; the skill lands in ("user", "skills",
    <user>) — the same namespace the /skills REST API and the frontend read —
    with the frontmatter parsed into the canonical name/description/content
    shape and the helper files bundled.
    """
    from app.core.constants import user_skills_ns
    from app.services import resources
    from app.services.agent import build_skill_publish_tool

    # What the agent writes with its file tools before publishing.
    draft = Path(settings.workspace_root) / "tester" / "scripts" / "deep-research"
    draft.mkdir(parents=True)
    (draft / "SKILL.md").write_text(
        "---\n"
        "name: deep-research\n"
        "description: Run rigorous multi-step web research.\n"
        "---\n\n"
        "# Deep Research\n\n"
        "1. Scope the question.\n"
    )
    (draft / "templates").mkdir()
    (draft / "templates" / "research-brief.md").write_text("# Research brief\n")

    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="publish_skill",
                        args={"skill_path": "/scripts/deep-research"},
                    )
                ],
            ),
            AIMessage(content="Skill published."),
        ]
    )
    agent = build_agent(
        checkpointer=memory_persistence.checkpointer,
        store=memory_persistence.store,
        mcp_tools=[build_skill_publish_tool()],
        model=model,
        system_prompt="test",
    )
    events = await collect_stream(agent, "tester", message="make me a skill")
    names = [e for e, _ in events]
    assert "tool_start" in names and "tool_end" in names, names
    tool_end = next(d for e, d in events if e == "tool_end")
    assert "Published skill 'deep-research'" in tool_end["output"]["content"], tool_end

    # The skill is in the user's store namespace, frontend-visible.
    ns = user_skills_ns("tester")
    skill = await resources.get_skill(memory_persistence.store, "deep-research", ns)
    assert skill is not None
    assert skill.content.startswith("---\nname: deep-research")
    assert "1. Scope the question." in skill.content
    assert [f.path for f in skill.files] == ["templates/research-brief.md"]
    assert [s.name for s in await resources.list_skills(memory_persistence.store, ns)] == [
        "deep-research"
    ]

    # Another user's namespace stays empty (per-user isolation).
    assert (
        await resources.get_skill(
            memory_persistence.store, "deep-research", user_skills_ns("someone-else")
        )
        is None
    )


async def test_agent_writes_skill_directly_into_skills_dir(memory_persistence, monkeypatch):
    """A skill the agent writes straight into skills/ persists to the store.

    Approach B: the workspace filesystem is the authoring surface. The agent
    calls write_file on /skills/<name>/SKILL.md during a run; the run-end
    writeback (sync_skills_to_store) stores it — no publish_skill dance, no
    tmp/ draft — and the skill is frontend-visible and survives future runs.
    """
    monkeypatch.setattr(settings, "execute_enabled", True)
    from app.core.constants import user_skills_ns
    from app.services import resources

    skill_md = (
        "---\n"
        "name: alpha\n"
        "description: Alpha skill, authored directly in skills/.\n"
        "---\n\n"
        "# Alpha\n\n"
        "1. Do the thing.\n"
    )
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="write_file",
                        args={"file_path": "/skills/alpha/SKILL.md", "content": skill_md},
                    )
                ],
            ),
            AIMessage(content="Skill created."),
        ]
    )
    agent = build_agent(
        checkpointer=memory_persistence.checkpointer,
        store=memory_persistence.store,
        model=model,
        system_prompt="test",
    )
    events = await collect_stream(agent, "tester", message="make me a skill")
    names = [e for e, _ in events]
    assert "tool_start" in names and "tool_end" in names, names
    tool_end = next(d for e, d in events if e == "tool_end")
    assert tool_end["name"] == "write_file"
    assert not tool_end.get("is_error"), tool_end

    # The run-end writeback persisted the direct write into the store.
    ns = user_skills_ns("tester")
    skill = await resources.get_skill(memory_persistence.store, "alpha", ns)
    assert skill is not None
    assert skill.content == skill_md  # raw frontmatter preserved
    assert [s.name for s in await resources.list_skills(memory_persistence.store, ns)] == ["alpha"]

    # The materialized mirror on disk matches what the agent wrote.
    on_disk = Path(settings.workspace_root) / "tester" / "skills" / "alpha" / "SKILL.md"
    assert on_disk.read_text() == skill_md


async def test_publish_skill_overwrite_semantics(memory_persistence):
    """Duplicate publish errors without overwrite; overwrite=true replaces."""
    from app.services.agent import build_skill_publish_tool

    tool = build_skill_publish_tool()
    # Outside a run the tool resolves to the "anonymous" user dir.
    draft = Path(settings.workspace_root) / "anonymous" / "tmp" / "demo"
    draft.mkdir(parents=True)
    (draft / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo skill.\n---\n\nBody.\n"
    )

    first = await tool.ainvoke({"skill_path": "/tmp/demo"})
    assert "Published skill 'demo-skill'" in first, first

    second = await tool.ainvoke({"skill_path": "/tmp/demo"})
    assert "already exists" in second, second

    (draft / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo skill, v2.\n---\n\nBody v2.\n"
    )
    third = await tool.ainvoke({"skill_path": "/tmp/demo", "overwrite": True})
    assert "Published skill 'demo-skill'" in third, third

    from app.core.constants import user_skills_ns
    from app.services import resources

    skill = await resources.get_skill(
        memory_persistence.store, "demo-skill", user_skills_ns("anonymous")
    )
    assert skill is not None and "Body v2." in skill.content


async def test_publish_skill_rejects_malformed_drafts(memory_persistence):
    """Missing folder / SKILL.md / frontmatter return errors instead of raising."""
    from app.services.agent import build_skill_publish_tool

    tool = build_skill_publish_tool()
    draft = Path(settings.workspace_root) / "anonymous" / "tmp" / "broken"
    draft.mkdir(parents=True)

    out = await tool.ainvoke({"skill_path": "/tmp/broken"})
    assert "no SKILL.md" in out, out

    (draft / "SKILL.md").write_text("no frontmatter here\n")
    out = await tool.ainvoke({"skill_path": "/tmp/broken"})
    assert "frontmatter" in out, out

    (draft / "SKILL.md").write_text("---\nname: broken\n---\n\nBody.\n")
    out = await tool.ainvoke({"skill_path": "/tmp/broken"})
    assert "description" in out, out

    out = await tool.ainvoke({"skill_path": "/tmp/does-not-exist"})
    assert "no folder" in out, out
