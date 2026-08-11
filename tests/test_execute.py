"""Offline tests for the execute (shell) tool — no network, no API key.

The `execute` tool is registered by deepagents' FilesystemMiddleware but only
works when the agent's backend supports execution (LocalShellBackend, opt-in
via EXECUTE_ENABLED). With the default StateBackend it returns a
"Execution not available" error instead of running commands.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from app.core import config
from app.main import create_app
from app.services.agent import build_agent
from app.services.chat import agent_stream

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


def execute_scripted_model() -> Scripted:
    return Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-1", name="execute", args={"command": "echo hi"})],
            ),
            AIMessage(content="Command output above."),
        ]
    )


def build_execute_agent():
    return build_agent(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        model=execute_scripted_model(),
        system_prompt="test",
    )


def parse_sse_chunk(chunk: str) -> tuple[str, dict]:
    ev, _, rest = chunk.partition("\n")
    return ev.removeprefix("event: "), json.loads(rest.removeprefix("data: ").strip())


async def collect_stream(agent, username, **kwargs) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async for chunk in agent_stream(agent, username, **kwargs):
        events.append(parse_sse_chunk(chunk))
    return events


async def tool_output(events) -> str:
    tool_end = next(d for e, d in events if e == "tool_end")
    content = tool_end["output"]["content"]
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    return content


async def test_execute_disabled_by_default_errors(monkeypatch):
    """EXECUTE_ENABLED=false: tool exists but refuses to run commands."""
    monkeypatch.setattr(config.settings, "execute_enabled", False)
    agent = build_execute_agent()
    events = await collect_stream(agent, "tester", message="run echo hi")
    out = await tool_output(events)
    assert "Execution not available" in out, out


async def test_execute_enabled_runs_command(monkeypatch):
    """EXECUTE_ENABLED=true: the execute tool runs the command (echo hi)."""
    monkeypatch.setattr(config.settings, "execute_enabled", True)
    agent = build_execute_agent()
    events = await collect_stream(agent, "tester", message="run echo hi")
    out = await tool_output(events)
    assert "hi" in out, out


async def test_execute_timeout_cap_enforced(monkeypatch):
    """Per-command timeouts above EXECUTE_MAX_TIMEOUT are rejected."""
    monkeypatch.setattr(config.settings, "execute_enabled", True)
    monkeypatch.setattr(config.settings, "execute_max_timeout", 5)
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="execute",
                        args={"command": "echo hi", "timeout": 9999},
                    )
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    agent = build_agent(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        model=model,
        system_prompt="test",
    )
    events = await collect_stream(agent, "tester", message="run echo hi")
    out = await tool_output(events)
    assert "exceeds maximum allowed" in out, out


async def test_health_reports_execute():
    from app.core import config as app_config
    from app.core.database import persistence

    app_config.settings.database_uri = None
    await persistence.start()
    try:
        app = create_app(agent=build_execute_agent())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            r = await http.get("/health")
            assert r.status_code == 200
            body = r.json()["execute"]
            assert body["enabled"] is config.settings.execute_enabled
            assert body["max_timeout"] == config.settings.execute_max_timeout
    finally:
        await persistence.stop()
