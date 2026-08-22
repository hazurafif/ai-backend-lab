"""Offline tests for the OpenUI generative-UI instructions tool.

Covers the TrueForge-style deferred-instructions pattern:
- the `get_openui_instructions` tool is always registered on agents,
- it returns the full OpenUI authoring instruction (fencing, syntax,
  components, built-ins, rules) on demand,
- the default system prompt tells the agent to load it before emitting
  any ```openui block,
- a full run: the scripted agent loads the instructions via the tool, then
  streams an ```openui fence that survives the SSE pipeline untouched.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, ToolCall
from test_smoke import Scripted, collect_stream

from app.core.config import settings
from app.core.constants import DEFAULT_SYSTEM_PROMPT
from app.core.database import persistence
from app.services.agent import BUILTIN_PSEUDO_TOOLS, build_agent, build_extra_tools
from app.services.openui import OPENUI_INSTRUCTIONS, build_openui_instructions_tool

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest_asyncio.fixture
async def memory_persistence():
    """Force in-memory checkpointer/store and start the app singleton."""
    settings.database_uri = None
    await persistence.start()
    yield persistence
    await persistence.stop()


def test_instructions_tool_registered_by_default():
    """`get_openui_instructions` is a core capability, always registered."""
    names = [t.name for t in build_extra_tools()]
    assert "get_openui_instructions" in names


def test_instructions_tool_selectable_by_name():
    """Per-agent tool lists can opt into the tool by name (like web_search)."""
    assert "get_openui_instructions" in BUILTIN_PSEUDO_TOOLS


async def test_instructions_tool_payload():
    """The tool returns the full authoring instruction on demand."""
    tool = build_openui_instructions_tool()
    out = await tool.ainvoke({})

    assert "```openui" in out, "fencing rule missing"
    assert "root = Stack" in out, "entry-point rule missing"
    assert "positional" in out.lower(), "positional-args rule missing"
    assert "@Count" in out and "@Each" in out, "built-in functions missing"
    assert "BarChart" in out and "Table" in out and "Form" in out, "components missing"
    assert "@ToAssistant" in out, "action steps missing"
    # The payload is the same constant the REACT frontend renders against.
    assert out == OPENUI_INSTRUCTIONS


def test_system_prompt_tells_agent_to_load_instructions():
    """The default prompt defers the instructions to the tool call."""
    assert "get_openui_instructions" in DEFAULT_SYSTEM_PROMPT
    assert "```openui" in DEFAULT_SYSTEM_PROMPT
    # The hint must not duplicate the full instruction.
    assert "COMPONENT SIGNATURES" not in DEFAULT_SYSTEM_PROMPT


def test_instructions_payload_is_consistent():
    """Structural sanity: every section header lands in the payload."""
    for marker in (
        "FENCING",
        "SYNTAX RULES",
        "COMPONENT SIGNATURES",
        "BUILT-IN FUNCTIONS",
        "REACTIVE VARIABLES",
        "HOISTING AND STREAMING",
        "EXAMPLES",
        "IMPORTANT RULES",
        "FINAL VERIFICATION",
        "USER INTERACTION CHECKLIST",
    ):
        assert marker in OPENUI_INSTRUCTIONS, f"missing section {marker}"


async def test_agent_loads_instructions_then_emits_openui_block(memory_persistence):
    """Full loop: tool call → instruction text → ```openui fence in the SSE stream.

    The scripted model calls `get_openui_instructions`, receives the payload,
    then answers with a fenced openui block; the block must arrive in the
    message_delta events unchanged.
    """
    fence = '```openui\nroot = Stack([t])\nt = TextContent("Hello")\n```'
    model = Scripted(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(id="call-openui", name="get_openui_instructions", args={})],
            ),
            AIMessage(content=fence),
        ]
    )
    agent = build_agent(
        checkpointer=memory_persistence.checkpointer,
        store=memory_persistence.store,
        mcp_tools=[],
        extra_tools=[build_openui_instructions_tool()],
        model=model,
        system_prompt="test",
    )
    events = await collect_stream(agent, "tester", message="show me a dashboard")

    names = [e for e, _ in events]
    assert "tool_start" in names and "tool_end" in names, "tool lifecycle missing"
    tool_end = next(d for e, d in events if e == "tool_end")
    assert tool_end["name"] == "get_openui_instructions"
    assert "root = Stack" in tool_end["output"]["content"]

    text = "".join(d["delta"] for e, d in events if e == "message_delta")
    assert "```openui" in text, "fence lost in the SSE stream"
    assert "root = Stack([t])" in text
