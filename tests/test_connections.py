"""Offline tests for API connections (base URL + API key for the agent's model).

Covers:

  - /agent/connections CRUD via HTTP: admin-only mutations, readable by any
    user, api_key is write-only (never returned), update merges (omitted
    api_key keeps the stored key)
  - agent configs: `connection` field validation (unknown name -> 400),
    resolution of the builtin default agent to the 'default' connection
  - model building: the registry constructs the chat model with the stored
    base_url + api_key (ChatOpenAI attributes, no network at construction)
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
from app.services import agent_configs, connections
from app.services.agent import AgentRegistry, build_backend
from app.services.agent_configs import AgentSpec

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)

BASE_URL = "https://api.example.com/v1"
API_KEY = "sk-test-connection"  # gitguardian:ignore


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


def make_app(persistence):
    """App with a registry that never touches the network (model factory)."""
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
        model_factory=lambda m, t: _DummyModel(),
    )
    return create_app(agent_registry=registry)


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


async def client_for(app, username: str, role: str = "user") -> httpx.AsyncClient:
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def connection_payload(name: str, **overrides) -> dict:
    payload = {"name": name, "base_url": BASE_URL, "api_key": API_KEY}
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# /agent/connections CRUD
# ---------------------------------------------------------------------------


async def test_connection_crud(persistence):
    app = make_app(persistence)
    async with (
        app.router.lifespan_context(app),
        await client_for(app, "boss", role="admin") as client,
    ):
        # create
        r = await client.post("/agent/connections", json=connection_payload("openai"))
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["name"] == "openai"
        assert body["base_url"] == BASE_URL
        assert body["has_api_key"] is True
        assert "api_key" not in body

        # duplicate -> 409
        r = await client.post("/agent/connections", json=connection_payload("openai"))
        assert r.status_code == 409, r.text

        # list + get never leak the key
        r = await client.get("/agent/connections")
        assert r.status_code == 200 and len(r.json()) == 1
        assert "api_key" not in r.json()[0]
        r = await client.get("/agent/connections/openai")
        assert r.status_code == 200 and r.json()["base_url"] == BASE_URL
        assert "api_key" not in r.json()

        # update: change base_url only, api_key stays stored
        r = await client.put(
            "/agent/connections/openai",
            json={"base_url": "https://alt.example.com/v1"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["base_url"] == "https://alt.example.com/v1"
        assert r.json()["has_api_key"] is True
        stored = await connections.load_connection(persistence.store, "openai")
        assert stored == {
            "base_url": "https://alt.example.com/v1",
            "api_key": API_KEY,
        }

        # update: rotate the api_key, keep the base_url
        r = await client.put("/agent/connections/openai", json={"api_key": "sk-rotated"})
        assert r.status_code == 200, r.text
        stored = await connections.load_connection(persistence.store, "openai")
        assert stored["api_key"] == "sk-rotated"
        assert stored["base_url"] == "https://alt.example.com/v1"

        # update with no fields -> 400
        r = await client.put("/agent/connections/openai", json={})
        assert r.status_code == 400, r.text

        # delete
        r = await client.delete("/agent/connections/openai")
        assert r.status_code == 204
        r = await client.get("/agent/connections/openai")
        assert r.status_code == 404
        r = await client.delete("/agent/connections/openai")
        assert r.status_code == 404
        r = await client.put("/agent/connections/openai", json={"base_url": BASE_URL})
        assert r.status_code == 404


async def test_connection_permissions(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app):
        # non-admin cannot create/update/delete, but can list/get
        async with await client_for(app, "alice") as alice:
            assert (
                await alice.post("/agent/connections", json=connection_payload("openai"))
            ).status_code == 403
            assert (
                await alice.put("/agent/connections/openai", json={"base_url": BASE_URL})
            ).status_code == 403
            assert (await alice.delete("/agent/connections/openai")).status_code == 403
            assert (await alice.get("/agent/connections")).status_code == 200

        # admin can create
        async with await client_for(app, "boss", role="admin") as admin:
            assert (
                await admin.post("/agent/connections", json=connection_payload("openai"))
            ).status_code == 201

        # non-admin sees it but cannot touch it
        async with await client_for(app, "bob") as bob:
            r = await bob.get("/agent/connections/openai")
            assert r.status_code == 200 and r.json()["name"] == "openai"
            assert (await bob.delete("/agent/connections/openai")).status_code == 403

        # schema validation: bad name / empty api_key
        async with await client_for(app, "boss2", role="admin") as admin:
            r = await admin.post("/agent/connections", json=connection_payload("Bad Name"))
            assert r.status_code == 422, r.text
            r = await admin.post("/agent/connections", json=connection_payload("x", api_key=""))
            assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# agent configs reference connections
# ---------------------------------------------------------------------------


async def test_agent_config_validates_connection(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app):
        async with await client_for(app, "alice") as alice:
            # unknown connection -> 400
            r = await alice.post(
                "/agents",
                json={
                    "name": "research",
                    "model": "openai:gpt-4o-mini",
                    "connection": "ghost",
                },
            )
            assert r.status_code == 400, r.text

        # admin creates the connection
        async with await client_for(app, "boss", role="admin") as admin:
            assert (
                await admin.post("/agent/connections", json=connection_payload("openai"))
            ).status_code == 201

        # now the agent config accepts it, and the output carries it
        async with await client_for(app, "alice") as alice:
            r = await alice.post(
                "/agents",
                json={
                    "name": "research",
                    "model": "openai:gpt-4o-mini",
                    "connection": "openai",
                },
            )
            assert r.status_code == 201, r.text
            assert r.json()["connection"] == "openai"

            # the builtin default agent resolves to 'default' only when it exists
            r = await alice.get("/agents/default")
            assert r.status_code == 200 and r.json()["connection"] is None


async def test_default_agent_uses_default_connection(persistence):
    app = make_app(persistence)
    async with app.router.lifespan_context(app):
        async with await client_for(app, "boss", role="admin") as admin:
            assert (
                await admin.post("/agent/connections", json=connection_payload("default"))
            ).status_code == 201

        spec = await agent_configs.load_spec(persistence.store, "default", "alice")
        assert spec is not None and spec.connection == "default"


# ---------------------------------------------------------------------------
# model building from a stored connection
# ---------------------------------------------------------------------------


async def test_resolve_model_uses_stored_connection(persistence):
    """The registry builds ChatOpenAI with the stored base_url + api_key."""
    await connections.create_connection(
        persistence.store, ConnectionIn(name="openai", base_url=BASE_URL, api_key=API_KEY)
    )
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
    )
    spec = AgentSpec(
        name="openai-agent",
        model="openai:gpt-4o-mini",
        system_prompt=None,
        skills=None,
        tools=None,
        temperature=None,
        interrupt_on=None,
        thinking=None,
        connection="openai",
    )
    model = await registry._resolve_model(spec)
    assert model.openai_api_base == BASE_URL  # type: ignore[attr-defined]
    assert model.openai_api_key.get_secret_value() == API_KEY  # type: ignore[attr-defined]
    assert model.model_name == "gpt-4o-mini"

    # unknown connection -> ValueError
    spec.connection = "ghost"
    with pytest.raises(ValueError, match="ghost"):
        await registry._resolve_model(spec)


async def test_resolve_model_keeps_env_behavior_without_connection(persistence):
    """Without a connection and without temperature, the model string is passed through."""
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=build_backend(store=persistence.store),
    )
    spec = agent_configs.default_spec()
    assert await registry._resolve_model(spec) == spec.model
