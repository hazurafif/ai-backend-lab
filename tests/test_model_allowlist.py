"""Offline tests for the global model allowlist (admin-managed, role-based).

A single allowlist applies to every `user`-role account (guests included);
admins are never restricted. Covers:

  - store (app_settings): get/set/clear/is_restricted/is_model_allowed,
    effective_for_role (admin vs user)
  - endpoints: admin GET/PUT/DELETE /allowed-models, the user-facing
    GET /users/me/allowed-models, 403 for non-admins
  - enforcement: user-scoped agent configs reject models outside the
    allowlist (403) while admin-created configs and unrestricted states keep
    working; chat on a restricted model stops with 403 for users, not admins
"""

from __future__ import annotations

import httpx
import pytest

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services import model_allowlist

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest.fixture(autouse=True)
def _offline_env():
    config.settings.database_uri = None


async def _client(username: str, role: str = "user", app=None):
    """A client bound to `app` (or a fresh one), sharing one lifespan+store.

    The lifespan re-initializes the in-memory stores on entry, so every
    client of a test must share the same app/lifespan — otherwise data set
    through one client is wiped by the next client's startup.
    """
    app = app or create_app()
    if not getattr(app.state, "_lifespan_started", False):
        await app.router.lifespan_context(app).__aenter__()
        app.state._lifespan_started = True
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


def _agent_payload(name: str, model: str, **overrides) -> dict:
    payload = {
        "name": name,
        "model": model,
        "description": "test agent",
        "system_prompt": f"You are the {name} agent.",
        "skills": None,
        "tools": None,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


async def test_allowlist_store_semantics():
    await database.persistence.start()
    try:
        # unset -> unrestricted
        assert await model_allowlist.get_allowed_models() == []
        assert await model_allowlist.is_restricted() is False
        assert await model_allowlist.is_model_allowed("anything") is True

        # set -> only listed models
        await model_allowlist.set_allowed_models(["openai:gpt-4o-mini"])
        assert await model_allowlist.is_restricted() is True
        assert await model_allowlist.is_model_allowed("openai:gpt-4o-mini") is True
        assert await model_allowlist.is_model_allowed("gemini-2.5-pro") is False

        # empty list -> allow nothing
        await model_allowlist.set_allowed_models([])
        assert await model_allowlist.is_model_allowed("openai:gpt-4o-mini") is False

        # clear -> unrestricted again
        assert await model_allowlist.clear_allowed_models() is True
        assert await model_allowlist.clear_allowed_models() is False
        assert await model_allowlist.is_restricted() is False

        # role semantics: admins bypass, users (and guests) don't
        await model_allowlist.set_allowed_models(["openai:gpt-4o-mini"])
        assert model_allowlist.role_allows_all("admin") is True
        assert model_allowlist.role_allows_all("user") is False
        assert model_allowlist.role_allows_all(None) is False  # guests
        eff = await model_allowlist.effective_for_role("admin")
        assert eff == {"restricted": False, "models": ["openai:gpt-4o-mini"]}
        eff = await model_allowlist.effective_for_role("user")
        assert eff == {"restricted": True, "models": ["openai:gpt-4o-mini"]}
    finally:
        await database.persistence.stop()


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


async def test_admin_allowlist_crud():
    admin = await _client("root", role="admin")
    async with admin:
        # unset -> unrestricted
        r = await admin.get("/allowed-models")
        assert r.status_code == 200 and r.json() == {"restricted": False, "models": []}

        # set -> restricted view
        r = await admin.put(
            "/allowed-models",
            json={"models": ["openai:gpt-4o-mini", "gemini-2.5-pro"]},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {
            "restricted": True,
            "models": ["openai:gpt-4o-mini", "gemini-2.5-pro"],
        }

        # GET reflects it
        r = await admin.get("/allowed-models")
        assert r.json() == {"restricted": True, "models": ["openai:gpt-4o-mini", "gemini-2.5-pro"]}

        # empty list -> allow nothing
        r = await admin.put("/allowed-models", json={"models": []})
        assert r.json() == {"restricted": True, "models": []}

        # DELETE -> unrestricted
        r = await admin.delete("/allowed-models")
        assert r.status_code == 204, r.text
        r = await admin.get("/allowed-models")
        assert r.json() == {"restricted": False, "models": []}


async def test_allowlist_endpoint_permissions_and_me_view():
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        r = await admin.put("/allowed-models", json={"models": ["gemini-2.5-pro"]})
        assert r.status_code == 200, r.text

    # users cannot set (or read) the global allowlist
    user = await _client("alice", app=app)
    async with user:
        r = await user.put("/allowed-models", json={"models": ["x"]})
        assert r.status_code == 403, r.text
        r = await user.get("/allowed-models")
        assert r.status_code == 403, r.text

        # but they read their own restriction via /users/me
        r = await user.get("/users/me/allowed-models")
        assert r.json() == {"restricted": True, "models": ["gemini-2.5-pro"]}

    # admins are never restricted in their own view
    admin2 = await _client("root", role="admin", app=app)
    async with admin2:
        r = await admin2.get("/users/me/allowed-models")
        assert r.json() == {"restricted": False, "models": ["gemini-2.5-pro"]}


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------


async def test_agent_config_create_respects_allowlist():
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        r = await admin.put("/allowed-models", json={"models": ["openai:gpt-4o-mini"]})
        assert r.status_code == 200, r.text

    alice = await _client("alice", app=app)
    async with alice:
        # disallowed model -> 403
        r = await alice.post("/agents", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 403, r.text
        assert "not allowed" in r.json()["detail"]

        # allowed model -> 201
        r = await alice.post("/agents", json=_agent_payload("research", "openai:gpt-4o-mini"))
        assert r.status_code == 201, r.text

        # update to a disallowed model -> 403 (config stays intact)
        r = await alice.put("/agents/research", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 403, r.text
        r = await alice.get("/agents/research")
        assert r.json()["model"] == "openai:gpt-4o-mini"


async def test_admins_bypass_allowlist():
    """Admins may create user-scoped agents with any model."""
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        await admin.put("/allowed-models", json={"models": ["openai:gpt-4o-mini"]})
        r = await admin.post("/agents", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 201, r.text
        # global agents are also unaffected
        r = await admin.post(
            "/agents", json=_agent_payload("shared", "gemini-2.5-pro", scope="global")
        )
        assert r.status_code == 201, r.text


async def test_unrestricted_state_allows_everything():
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        # never set -> no restriction
        alice = await _client("alice", app=app)
        r = await alice.post("/agents", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 201, r.text
        r = await alice.get("/users/me/allowed-models")
        assert r.json() == {"restricted": False, "models": []}


async def test_chat_blocked_on_restricted_model_for_users():
    """Users chatting on a model outside the allowlist get 403 (default agent included)."""
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
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

        # default agent model is openai:deepseek-v4-flash — not allowed
        await admin.put("/allowed-models", json={"models": ["gemini-2.5-pro"]})

    alice = await _client("alice", app=app)
    async with alice:
        r = await alice.post("/chat", json={"message": "hello"})
        assert r.status_code == 403, r.text
        assert "deepseek-v4-flash" in r.json()["detail"]

        # an allowed model is not gated at config time
        r = await alice.post("/agents", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 201, r.text

    # an admin chats normally on the same default model
    admin2 = await _client("root", role="admin", app=app)
    async with admin2:
        r = await admin2.post("/chat", json={"message": "hello"})
        assert r.status_code == 200, r.text
