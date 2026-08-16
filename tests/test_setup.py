"""Offline tests for the setup / onboarding flow (per-user data, admin LLM).

Verifies:

  - GET /users/me/setup reports the state: admin-managed llm connection
    (read-only, masked) + effective model + the user's own preferences and
    MCP tool servers
  - POST /users/me/onboarding saves per-user preferences + MCP tool servers
    (idempotent) and never touches connections
  - /connections stays admin-only: regular users get 403 and cannot read or
    write credentials
  - the agent model build resolves the admin-managed default llm connection
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from langchain_core.language_models import BaseChatModel

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services import connections as connection_service
from app.services.agent import AgentRegistry
from app.services.agent_configs import AgentSpec

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    config.settings.database_uri = None
    monkeypatch.setattr(config.settings, "model", None)


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


async def _app_client(
    username: str = "alice", role: str = "user"
) -> tuple[httpx.AsyncClient, object]:
    app = create_app()
    await app.router.lifespan_context(app).__aenter__()
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ),
        app,
    )


def llm_payload(name: str = "zen", **overrides) -> dict:
    body = {
        "name": name,
        "kind": "llm",
        "base_url": "https://api.example.com/v1",
        "api_token": "sk-admin-key-1234",
        "extra": {"model": "openai:deepseek-v4-flash"},
        "is_default": True,
    }
    body.update(overrides)
    return body


async def test_setup_fresh_state():
    client, _app = await _app_client()
    async with client:
        r = await client.get("/users/me/setup")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["completed"] is False
        assert body["llm_connection"] is None
        assert body["model"] is None
        assert body["mcp_servers"] == []
        assert body["preferences"] == {
            "enable_search": config.settings.searxng_enabled,
            "hide_reasoning": False,
            "hide_tool_calls": False,
        }


async def test_onboarding_saves_per_user_data_only():
    client, _app = await _app_client()
    async with client:
        r = await client.post(
            "/users/me/onboarding",
            json={
                "preferences": {"hide_reasoning": True, "hide_tool_calls": True},
                "mcp_servers": [
                    {
                        "name": "weather",
                        "transport": "streamable_http",
                        "url": "http://localhost:8090/mcp",
                    }
                ],
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        # connections are admin-managed: still none for the user
        assert body["completed"] is False
        assert body["llm_connection"] is None
        # preferences persisted per user
        assert body["preferences"]["hide_reasoning"] is True
        assert body["preferences"]["hide_tool_calls"] is True
        # MCP servers persisted per user
        assert [s["name"] for s in body["mcp_servers"]] == ["weather"]

        prefs = await database.persistence.preferences.get_all("alice")
        assert prefs == {"hide_reasoning": True, "hide_tool_calls": True}
        servers = await database.persistence.store.asearch(("user", "mcp_servers", "alice"))
        assert [it.key for it in servers] == ["weather"]
        # no connection rows were created for the user
        assert await database.persistence.connections.list() == []

        # idempotent: re-running updates, never duplicates
        r = await client.post(
            "/users/me/onboarding",
            json={
                "preferences": {"hide_reasoning": False},
                "mcp_servers": [
                    {
                        "name": "weather",
                        "transport": "streamable_http",
                        "url": "http://localhost:8090/mcp",
                    }
                ],
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["preferences"]["hide_reasoning"] is False
        assert [s["name"] for s in r.json()["mcp_servers"]] == ["weather"]


async def test_setup_reflects_admin_connection():
    """Once the admin saves a default llm connection, setup reports it."""
    client, _app = await _app_client()
    async with client:
        await database.persistence.connections.create(llm_payload())
        await connection_service.refresh_resolved_connections()
        r = await client.get("/users/me/setup")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["completed"] is True
        assert body["llm_connection"]["name"] == "zen"
        assert body["llm_connection"]["api_token"] == "sk-a…1234"  # masked, read-only
        assert body["llm_connection"]["has_token"] is True
        assert body["model"] == "openai:deepseek-v4-flash"


async def test_connections_admin_only():
    """Regular users can neither read nor write connections (403)."""
    client, _app = await _app_client()  # role=user
    async with client:
        r = await client.get("/connections")
        assert r.status_code == 403
        r = await client.post("/connections", json=llm_payload())
        assert r.status_code == 403
        r = await client.get("/connections/zen")
        assert r.status_code == 403
        r = await client.put("/connections/zen", json=llm_payload())
        assert r.status_code == 403
        r = await client.delete("/connections/zen")
        assert r.status_code == 403


async def test_admin_can_manage_connections():
    client, _app = await _app_client(username="boss", role="admin")
    async with client:
        r = await client.post("/connections", json=llm_payload())
        assert r.status_code == 201, r.text
        r = await client.get("/connections")
        assert [c["name"] for c in r.json()] == ["zen"]


async def test_agent_build_uses_admin_connection(persistence):
    """The agent's chat model resolves the admin-managed llm connection."""
    await persistence.connections.create(llm_payload())
    await connection_service.refresh_resolved_connections()
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=None,
    )
    spec = AgentSpec(
        name="default",
        model=None,
        system_prompt=None,
        skills=None,
        tools=None,
        temperature=None,
        interrupt_on=None,
        thinking=None,
        builtin=True,
    )
    model = registry._resolve_model(spec)
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "openai_api_base", None) == "https://api.example.com/v1"
    assert model.openai_api_key.get_secret_value() == "sk-admin-key-1234"  # type: ignore[attr-defined]
    assert model.model_name == "deepseek-v4-flash"
