"""Offline tests for the `thinking` (reasoning effort) agent-config field.

Covers:

  - /agents CRUD: thinking level persists through create/update, invalid
    levels -> 422, builtin default has thinking=None
  - model building: `reasoning_effort` is passed to OpenAI-compatible models
    (plain and via a stored connection)
  - fingerprint: thinking participates in the graph cache key
"""

from __future__ import annotations

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
from app.schema.connection_schema import ConnectionIn
from app.services import connections
from app.services.agent import AgentRegistry, build_backend
from app.services.agent_configs import AgentSpec

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)

BASE_URL = "https://api.example.com/v1"
API_KEY = "sk-test-thinking"  # gitguardian:ignore


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


class _DummyModel(BaseChatModel):
    """Minimal chat model returned by the model factory (never invoked)."""

    responses: list[AIMessage] = Field(default_factory=lambda: [AIMessage(content="ok")])
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "dummy"

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
        return self


def make_app(persistence):
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
        model_factory=lambda m, t: _DummyModel(),
    )
    return create_app(agent_registry=registry)


async def client_for(app, username: str, role: str = "user") -> httpx.AsyncClient:
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def agent_payload(name: str, **overrides) -> dict:
    payload = {"name": name, "model": "openai:gpt-5.2"}
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# /agents CRUD
# ---------------------------------------------------------------------------


async def test_thinking_crud(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app), await client_for(app, "alice") as client:
        # create with thinking=xhigh (extra high)
        r = await client.post("/agents", json=agent_payload("coder", thinking="xhigh"))
        assert r.status_code == 201, r.text
        assert r.json()["thinking"] == "xhigh"

        # get + list carry it; builtin default has none
        assert (await client.get("/agents/coder")).json()["thinking"] == "xhigh"
        assert (await client.get("/agents/default")).json()["thinking"] is None
        agents = (await client.get("/agents")).json()
        assert {a["name"]: a["thinking"] for a in agents} == {"default": None, "coder": "xhigh"}

        # update to max
        r = await client.put("/agents/coder", json=agent_payload("coder", thinking="max"))
        assert r.status_code == 200 and r.json()["thinking"] == "max"

        # invalid level -> 422
        r = await client.post("/agents", json=agent_payload("bad", thinking="extra-high"))
        assert r.status_code == 422, r.text
        r = await client.post("/agents", json=agent_payload("bad", thinking="maxx"))
        assert r.status_code == 422, r.text

        # test endpoint reports it
        r = await client.post("/agents/coder/test")
        assert r.status_code == 200 and r.json()["thinking"] == "max"


# ---------------------------------------------------------------------------
# model building
# ---------------------------------------------------------------------------


def _spec(thinking: str | None, **overrides) -> AgentSpec:
    values: dict[str, Any] = {
        "name": "coder",
        "model": overrides.pop("model", "openai:gpt-5.2"),
        "system_prompt": None,
        "skills": None,
        "tools": None,
        "temperature": None,
        "interrupt_on": None,
        "thinking": thinking,
        "connection": None,
    }
    values.update(overrides)
    return AgentSpec(**values)


async def test_resolve_model_passes_reasoning_effort(persistence, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # gitguardian:ignore
    monkeypatch.setenv("OPENAI_BASE_URL", BASE_URL)
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
    )
    model = await registry._resolve_model(_spec("xhigh"))
    assert model.reasoning_effort == "xhigh"  # type: ignore[attr-defined]
    assert model.model_name == "gpt-5.2"

    # combined with temperature (gpt-4o family keeps the top-level param)
    model = await registry._resolve_model(_spec("max", temperature=0.3, model="openai:gpt-4o-mini"))
    assert model.reasoning_effort == "max"  # type: ignore[attr-defined]
    assert model.temperature == 0.3

    # without thinking (and without temperature) the string passes through
    assert await registry._resolve_model(_spec(None)) == "openai:gpt-5.2"


async def test_resolve_model_thinking_with_connection(persistence):
    """thinking + stored connection: both kwargs reach ChatOpenAI."""
    await connections.create_connection(
        persistence.store, ConnectionIn(name="zen", base_url=BASE_URL, api_key=API_KEY)
    )
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
    )
    model = await registry._resolve_model(_spec("high", connection="zen"))
    assert model.reasoning_effort == "high"  # type: ignore[attr-defined]
    assert model.openai_api_base == BASE_URL  # type: ignore[attr-defined]
    assert model.openai_api_key.get_secret_value() == API_KEY  # type: ignore[attr-defined]


async def test_thinking_changes_fingerprint(persistence):
    """Different thinking levels produce different graph cache keys."""
    plain = _spec(None)
    xhigh = _spec("xhigh")
    assert plain.fingerprint() != xhigh.fingerprint()
    assert xhigh.fingerprint() == _spec("xhigh").fingerprint()
