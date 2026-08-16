"""Offline tests for per-user model allowlists (admin-managed).

Covers:

  - store: get/set/clear/is_restricted/is_model_allowed semantics (memory mode)
  - endpoints: admin PUT/GET/DELETE /users/{username}/allowed-models, the
    user-facing GET /users/me/allowed-models, 403 for non-admins, 404 for
    unknown users
  - enforcement: user-scoped agent configs reject models outside the
    allowlist (403) while global (admin) configs and unrestricted users keep
    working; chat on a restricted model stops with 403
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
        store = database.persistence.store
        # unset -> unrestricted
        assert await model_allowlist.get_allowed_models(store, "alice") == []
        assert await model_allowlist.is_restricted(store, "alice") is False
        assert await model_allowlist.is_model_allowed(store, "alice", "anything") is True

        # set -> only listed models
        await model_allowlist.set_allowed_models(store, "alice", ["openai:gpt-4o-mini"])
        assert await model_allowlist.is_restricted(store, "alice") is True
        assert await model_allowlist.is_model_allowed(store, "alice", "openai:gpt-4o-mini") is True
        assert await model_allowlist.is_model_allowed(store, "alice", "gemini-2.5-pro") is False

        # per-user isolation
        assert await model_allowlist.is_model_allowed(store, "bob", "gemini-2.5-pro") is True

        # empty list -> allow nothing
        await model_allowlist.set_allowed_models(store, "alice", [])
        assert await model_allowlist.is_model_allowed(store, "alice", "openai:gpt-4o-mini") is False

        # clear -> unrestricted again
        assert await model_allowlist.clear_allowed_models(store, "alice") is True
        assert await model_allowlist.clear_allowed_models(store, "alice") is False
        assert await model_allowlist.is_restricted(store, "alice") is False
    finally:
        await database.persistence.stop()


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


async def test_admin_allowlist_crud():
    admin = await _client("root", role="admin")
    async with admin:
        await database.persistence.users.create_user(username="alice", hashed_password="x")
        # unknown user -> 404
        r = await admin.get("/users/ghost/allowed-models")
        assert r.status_code == 404, r.text
        r = await admin.put("/users/ghost/allowed-models", json={"models": ["x"]})
        assert r.status_code == 404, r.text

        # set -> restricted view
        r = await admin.put(
            "/users/alice/allowed-models",
            json={"models": ["openai:gpt-4o-mini", "gemini-2.5-pro"]},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {
            "restricted": True,
            "models": ["openai:gpt-4o-mini", "gemini-2.5-pro"],
        }

        # GET reflects it
        r = await admin.get("/users/alice/allowed-models")
        assert r.json()["restricted"] is True
        assert r.json()["models"] == ["openai:gpt-4o-mini", "gemini-2.5-pro"]

        # empty list -> allow nothing
        r = await admin.put("/users/alice/allowed-models", json={"models": []})
        assert r.json() == {"restricted": True, "models": []}

        # DELETE -> unrestricted
        r = await admin.delete("/users/alice/allowed-models")
        assert r.status_code == 204, r.text
        r = await admin.get("/users/alice/allowed-models")
        assert r.json() == {"restricted": False, "models": []}


async def test_allowlist_non_admin_forbidden_and_me_view():
    user = await _client("alice")
    async with user:
        # users cannot set (or even manage) allowlists
        r = await user.put("/users/alice/allowed-models", json={"models": ["x"]})
        assert r.status_code == 403, r.text
        r = await user.get("/users/alice/allowed-models")
        assert r.status_code == 403, r.text

        # but they read their own restriction (unrestricted here)
        r = await user.get("/users/me/allowed-models")
        assert r.status_code == 200 and r.json() == {"restricted": False, "models": []}

    # admin restricts, the user sees it via /users/me (same app/store)
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        await database.persistence.users.create_user(username="alice", hashed_password="x")
        r = await admin.put("/users/alice/allowed-models", json={"models": ["gemini-2.5-pro"]})
        assert r.status_code == 200, r.text
    user2 = await _client("alice", app=app)
    async with user2:
        r = await user2.get("/users/me/allowed-models")
        assert r.json() == {"restricted": True, "models": ["gemini-2.5-pro"]}


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------


async def test_agent_config_create_respects_allowlist():
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        await database.persistence.users.create_user(username="alice", hashed_password="x")
        await admin.put("/users/alice/allowed-models", json={"models": ["openai:gpt-4o-mini"]})

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

    # unrestricted user is unaffected
    bob = await _client("bob")
    async with bob:
        r = await bob.post("/agents", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 201, r.text


async def test_global_agents_not_gated_by_allowlist():
    """Admin-created global agents may use any model (admins manage them)."""
    admin = await _client("root", role="admin")
    async with admin:
        await database.persistence.users.create_user(username="alice", hashed_password="x")
        await admin.put("/users/alice/allowed-models", json={"models": ["openai:gpt-4o-mini"]})
        r = await admin.post(
            "/agents",
            json=_agent_payload("research", "gemini-2.5-pro", scope="global"),
        )
        assert r.status_code == 201, r.text


async def test_chat_blocked_on_restricted_model():
    """A user chatting on a model outside the allowlist gets 403 (default agent included)."""
    app = create_app()
    admin = await _client("root", role="admin", app=app)
    async with admin:
        await database.persistence.users.create_user(username="alice", hashed_password="x")
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
        await admin.put("/users/alice/allowed-models", json={"models": ["gemini-2.5-pro"]})

    alice = await _client("alice", app=app)
    async with alice:
        r = await alice.post("/chat", json={"message": "hello"})
        assert r.status_code == 403, r.text
        assert "deepseek-v4-flash" in r.json()["detail"]

        # the allowed model is not gated at config time
        r = await alice.post("/agents", json=_agent_payload("research", "gemini-2.5-pro"))
        assert r.status_code == 201, r.text

        # users/me reflects the restriction
        r = await alice.get("/users/me/allowed-models")
        assert r.json() == {"restricted": True, "models": ["gemini-2.5-pro"]}

    # an unrestricted user chats normally on the same default model
    bob = await _client("bob", app=app)
    async with bob:
        r = await bob.post("/chat", json={"message": "hello"})
        assert r.status_code == 200, r.text
