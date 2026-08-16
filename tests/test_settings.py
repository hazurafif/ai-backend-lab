"""Offline tests for DB-backed app settings (execute toggle + connection policy).

Covers:

  - AppSettingsStore CRUD (in-memory fallback)
  - effective values: DB overrides .env; env is the fallback
  - GET /settings (admin-only) + PUT /settings flips the execute tool and the
    connection fallback policy at runtime (registry rebuild)
  - DB-only connections: _resolve_model fails loudly without a default `llm`
    connection unless env fallback is enabled
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from langchain_core.language_models import BaseChatModel

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services import settings as settings_service
from app.services.agent import AgentRegistry

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


async def _admin_client():
    """Admin client with the app lifespan running (app.state.agents bound)."""
    app = create_app()
    await app.router.lifespan_context(app).__aenter__()
    await database.persistence.users.create_user(username="boss", hashed_password="x", role="admin")
    token = create_access_token(data={"sub": "boss"})
    return (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ),
        app,
    )


# ---------------------------------------------------------------------------
# store + effective values
# ---------------------------------------------------------------------------


async def test_store_crud(persistence):
    await persistence.settings.set("execute", {"enabled": True, "max_timeout": 60})
    assert await persistence.settings.get("execute") == {"enabled": True, "max_timeout": 60}
    rows = await persistence.settings.list()
    assert rows[0]["key"] == "execute"
    await persistence.settings.delete("execute")
    assert await persistence.settings.get("execute") is None


async def test_effective_values_db_overrides_env(persistence, monkeypatch):
    monkeypatch.setattr(config.settings, "execute_enabled", False)

    # No DB rows -> env defaults.
    await settings_service.refresh_app_settings()
    assert settings_service.execute_enabled() is False

    # DB row wins.
    await persistence.settings.set("execute", {"enabled": True, "max_timeout": 120})
    await settings_service.refresh_app_settings()
    assert settings_service.execute_enabled() is True
    assert settings_service.execute_max_timeout() == 120


# ---------------------------------------------------------------------------
# HTTP: GET/PUT /settings
# ---------------------------------------------------------------------------


async def test_settings_admin_only():
    app = create_app()
    await app.router.lifespan_context(app).__aenter__()
    await database.persistence.users.create_user(
        username="regular", hashed_password="x", role="user"
    )
    token = create_access_token(data={"sub": "regular"})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        r = await client.get("/settings")
        assert r.status_code == 403


async def test_get_settings_reports_source(persistence):
    client, _app = await _admin_client()
    async with client:
        r = await client.get("/settings")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["execute"]["enabled"] is config.settings.execute_enabled
        assert body["execute"]["source"] == "env"


async def test_put_settings_flips_execute_and_rebuilds(persistence):
    client, app = await _admin_client()
    async with client:
        r = await client.put("/settings", json={"execute": {"enabled": True, "max_timeout": 30}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["execute"]["enabled"] is True
        assert body["execute"]["max_timeout"] == 30
        assert body["execute"]["source"] == "db"

        # The filesystem backend was rebuilt: the per-user workspace backend
        # is the default in both modes (execute gated inside it) and the
        # default agent graph was rebuilt on top of it.
        assert app.state.backend is app.state.agents.backend
        from app.services.agent import UserShellBackend

        assert isinstance(app.state.agents.backend.default, UserShellBackend)

        # Partial update keeps unset fields.
        r = await client.put("/settings", json={"execute": {"enabled": False}})
        body = r.json()
        assert body["execute"]["enabled"] is False
        assert body["execute"]["max_timeout"] == 30
        from app.services.agent import UserShellBackend

        assert isinstance(app.state.agents.backend.default, UserShellBackend)
        # Execute is refused while the opt-in is off.
        resp = app.state.agents.backend.default.execute("echo hi")  # type: ignore[attr-defined]
        assert "Execution not available" in resp.output

        # Health reflects the DB value.
        r = await client.get("/health")
        assert r.json()["execute"]["enabled"] is False


async def test_put_settings_hitl(persistence):
    client, _app = await _admin_client()
    async with client:
        # Enable HITL for the builtin default agent: pause before `execute`.
        r = await client.put("/settings", json={"hitl": {"interrupt_on": {"execute": True}}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["hitl"] == {
            "interrupt_on": {"execute": True},
            "source": "db",
        }

        # The builtin default agent now builds with the DB interrupt_on.
        from app.services.agent_configs import default_spec

        assert default_spec().interrupt_on == {"execute": True}

        # Health reflects it.
        r = await client.get("/health")
        assert r.json()["interrupt_on"] == {"execute": True}

        # Disable again ({} = off).
        r = await client.put("/settings", json={"hitl": {"interrupt_on": {}}})
        assert r.json()["hitl"]["interrupt_on"] is None


# ---------------------------------------------------------------------------
# DB-only connections: no silent .env fallback
# ---------------------------------------------------------------------------


def _registry(persistence) -> AgentRegistry:
    return AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=None,
    )


async def test_resolve_model_requires_db_connection(persistence):
    await settings_service.refresh_app_settings()

    from app.services.agent_configs import AgentSpec

    spec = AgentSpec(
        name="default",
        model="openai:gpt-4o-mini",
        system_prompt=None,
        skills=None,
        tools=None,
        temperature=None,
        interrupt_on=None,
        thinking=None,
        builtin=True,
    )
    with pytest.raises(ValueError, match="No default 'llm' connection"):
        _registry(persistence)._resolve_model(spec)

    # No env fallback exists: even a saved model cannot build without a
    # connection (spec model + connection credentials are one unit).
    spec.model = "openai:gpt-4o-mini"
    with pytest.raises(ValueError, match="No default 'llm' connection"):
        _registry(persistence)._resolve_model(spec)


async def test_resolve_model_uses_db_connection_when_present(persistence):
    await persistence.connections.create(
        {
            "name": "zen",
            "kind": "llm",
            "base_url": "https://api.example.com/v1",
            "api_token": "sk-db-token",
            "extra": {},
            "is_default": True,
        }
    )
    await settings_service.refresh_app_settings()
    from app.services import connections as connection_service

    await connection_service.refresh_resolved_connections()

    from app.services.agent_configs import AgentSpec

    spec = AgentSpec(
        name="default",
        model="openai:gpt-4o-mini",
        system_prompt=None,
        skills=None,
        tools=None,
        temperature=None,
        interrupt_on=None,
        thinking=None,
        builtin=True,
    )
    model: BaseChatModel = _registry(persistence)._resolve_model(spec)
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "openai_api_base", None) == "https://api.example.com/v1"
    assert model.openai_api_key.get_secret_value() == "sk-db-token"  # type: ignore[attr-defined]


async def test_resolve_model_uses_connection_model_when_spec_has_none(persistence, monkeypatch):
    """No env model: the default llm connection's extra.model is used."""
    monkeypatch.setattr(config.settings, "model", None)
    await persistence.connections.create(
        {
            "name": "zen",
            "kind": "llm",
            "base_url": "https://api.example.com/v1",
            "api_token": "sk-db-token",
            "extra": {"model": "openai:deepseek-v4-flash"},
            "is_default": True,
        }
    )
    await settings_service.refresh_app_settings()
    from app.services import connections as connection_service

    await connection_service.refresh_resolved_connections()

    from app.services.agent_configs import AgentSpec

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
    model: BaseChatModel = _registry(persistence)._resolve_model(spec)
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "openai_api_base", None) == "https://api.example.com/v1"
    assert model.model_name == "deepseek-v4-flash"

    # An explicit spec model still wins over the connection's model.
    spec.model = "openai:gpt-4o-mini"
    model = _registry(persistence)._resolve_model(spec)
    assert model.model_name == "gpt-4o-mini"


async def test_resolve_model_requires_model_when_unconfigured(persistence, monkeypatch):
    """No model anywhere (env + connection) -> loud error, never a default."""
    monkeypatch.setattr(config.settings, "model", None)
    await settings_service.refresh_app_settings()
    from app.services import connections as connection_service

    await connection_service.refresh_resolved_connections()

    from app.services.agent_configs import AgentSpec

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
    with pytest.raises(ValueError, match="No model configured"):
        _registry(persistence)._resolve_model(spec)


async def test_app_starts_without_model_then_configures_at_runtime(persistence, monkeypatch):
    """No model configured: the app still starts; chats 503 with instructions.

    Saving a default llm connection (extra.model) afterwards makes the agent
    buildable on the next request — no restart needed.
    """
    monkeypatch.setattr(config.settings, "model", None)
    app = create_app()
    cm = app.router.lifespan_context(app)
    await cm.__aenter__()
    try:
        assert app.state.agent is None  # started fine, graph built lazily

        await database.persistence.users.create_user(
            username="boss", hashed_password="x", role="admin"
        )
        token = create_access_token(data={"sub": "boss"})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            r = await client.post("/chat", json={"message": "hi"})
            assert r.status_code == 503, r.text
            assert "No model configured" in r.json()["detail"]

        # Configure the default llm connection in-app (no restart): the next
        # resolve builds a working graph.
        await database.persistence.connections.create(
            {
                "name": "zen",
                "kind": "llm",
                "base_url": "https://api.example.com/v1",
                "api_token": "sk-db-token",
                "extra": {"model": "openai:deepseek-v4-flash"},
                "is_default": True,
            }
        )
        from app.services import connections as connection_service

        await connection_service.refresh_resolved_connections()
        app.state.agents.invalidate()
        graph = await app.state.agents.resolve("default", "anonymous")
        assert graph is not None
    finally:
        await cm.__aexit__(None, None, None)
