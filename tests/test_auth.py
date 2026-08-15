"""Offline tests for the auth routes: /register, /login, /users/me.

The users store is dict-backed here (no Postgres in tests), and the default
admin seeding is exercised by `test_default_admin_seeded_on_first_start`.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from conftest import TEST_PASSWORD

from app.core import config
from app.core.database import persistence
from app.core.security import create_access_token, get_password_hash, verify_password
from app.main import create_app

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest.fixture
def client():
    app = create_app()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def fresh_user_store():
    """Snapshot and restore the users store around each test."""
    store = persistence.users
    snapshot = {name: dict(user) for name, user in store._memory.items()}
    store._memory.clear()
    yield store
    store._memory.clear()
    store._memory.update(snapshot)


@pytest.mark.asyncio
async def test_register_creates_user(client, fresh_user_store):
    response = await client.post(
        "/register",
        json={
            "username": "alice",
            "password": TEST_PASSWORD,
            "email": "alice@example.com",
            "full_name": "Alice Example",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "alice"
    assert body["email"] == "alice@example.com"
    assert body["full_name"] == "Alice Example"
    assert body["disabled"] is False
    assert "hashed_password" not in body

    # New user can log in with the chosen password.
    login = await client.post("/login", data={"username": "alice", "password": TEST_PASSWORD})
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_defaults_to_user_role(client, fresh_user_store):
    # Registration always creates a regular `user`; a role field in the body
    # is ignored (never self-elevation).
    response = await client.post(
        "/register",
        json={"username": "alice", "password": TEST_PASSWORD, "role": "admin"},
    )
    assert response.status_code == 201
    assert response.json()["role"] == "user"

    stored = await persistence.users.get_user("alice")
    assert stored["role"] == "user"


@pytest.mark.asyncio
async def test_register_duplicate_username_conflicts(client, fresh_user_store):
    first = await client.post("/register", json={"username": "bob", "password": TEST_PASSWORD})
    assert first.status_code == 201

    second = await client.post("/register", json={"username": "bob", "password": "another-secret"})
    assert second.status_code == 409
    assert "already taken" in second.json()["detail"]


@pytest.mark.asyncio
async def test_register_rejects_short_password(client, fresh_user_store):
    response = await client.post("/register", json={"username": "carol", "password": "short"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_rejects_invalid_username(client, fresh_user_store):
    response = await client.post(
        "/register", json={"username": "bad name!", "password": TEST_PASSWORD}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_me_returns_profile(client, fresh_user_store):
    await client.post(
        "/register",
        json={
            "username": "alice",
            "password": TEST_PASSWORD,
            "email": "alice@example.com",
            "full_name": "Alice Example",
        },
    )
    login = await client.post("/login", data={"username": "alice", "password": TEST_PASSWORD})
    token = login.json()["access_token"]

    response = await client.get("/users/me/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "alice"
    assert body["email"] == "alice@example.com"
    assert body["full_name"] == "Alice Example"
    assert body["disabled"] is False
    assert "hashed_password" not in body


@pytest.mark.asyncio
async def test_me_rejects_unknown_user(client, fresh_user_store):
    # A valid JWT for a user that does not exist in the store -> 401.
    token = create_access_token(data={"sub": "ghost"})
    response = await client.get("/users/me/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_default_admin_seeded_on_first_start():
    """First start with an empty users store seeds admin/admin; later starts don't duplicate."""
    config.settings.database_uri = None
    await persistence.start()
    try:
        user = await persistence.users.get_user("admin")
        assert user is not None
        assert verify_password("admin", user["hashed_password"])
        assert user["full_name"] == "Admin"
        assert user["role"] == "admin"

        # Restarting must not create a second admin.
        await persistence.stop()
        await persistence.start()
        assert await persistence.users.count() == 1

        # The seeded username is reserved by the register API.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            response = await client.post(
                "/register", json={"username": "admin", "password": TEST_PASSWORD}
            )
            assert response.status_code == 409

        # admin/admin can log in and read /users/me.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            login = await client.post("/login", data={"username": "admin", "password": "admin"})
            assert login.status_code == 200
            token = login.json()["access_token"]
            me = await client.get("/users/me/", headers={"Authorization": f"Bearer {token}"})
            assert me.status_code == 200
            assert me.json()["username"] == "admin"
            assert me.json()["role"] == "admin"
    finally:
        await persistence.stop()


@pytest.mark.asyncio
async def test_existing_default_admin_promoted_on_start():
    """Upgrade path: an existing default-admin username is promoted to admin."""
    config.settings.database_uri = None
    await persistence.start()
    try:
        # Simulate a pre-0003 install where the default admin is a plain user.
        await persistence.users.update_user("admin", role="user")
        await persistence.ensure_default_admin()
        assert (await persistence.users.get_user("admin"))["role"] == "admin"
    finally:
        await persistence.stop()


# ---------------------------------------------------------------------------
# roles & permissions (admin user management)
# ---------------------------------------------------------------------------


async def _login(client, username: str, password: str) -> str:
    login = await client.post("/login", data={"username": username, "password": password})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


@pytest.mark.asyncio
async def test_regular_user_cannot_list_users(client, fresh_user_store):
    await client.post("/register", json={"username": "alice", "password": TEST_PASSWORD})
    token = await _login(client, "alice", TEST_PASSWORD)

    response = await client.get("/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert "Admin role required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_admin_lists_and_updates_users(client, fresh_user_store):
    await client.post("/register", json={"username": "alice", "password": TEST_PASSWORD})
    # Promote alice to admin first, then use her token for management.
    await persistence.users.update_user("alice", role="admin")
    token = await _login(client, "alice", TEST_PASSWORD)
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.get("/users", headers=headers)
    assert response.status_code == 200
    users = {u["username"]: u for u in response.json()}
    assert "alice" in users and users["alice"]["role"] == "admin"
    assert "hashed_password" not in users["alice"]

    # Promote bob, then disable him.
    await client.post("/register", json={"username": "bob", "password": TEST_PASSWORD})
    r = await client.patch("/users/bob", json={"role": "admin"}, headers=headers)
    assert r.status_code == 200 and r.json()["role"] == "admin"

    r = await client.patch("/users/bob", json={"disabled": True}, headers=headers)
    assert r.status_code == 200 and r.json()["disabled"] is True

    # Bob's existing token no longer works (account disabled).
    bob_token = await _login(client, "bob", TEST_PASSWORD)
    r = await client.get("/users/me/", headers={"Authorization": f"Bearer {bob_token}"})
    assert r.status_code == 403

    # Unknown user -> 404; empty body -> 422.
    r = await client.patch("/users/ghost", json={"disabled": True}, headers=headers)
    assert r.status_code == 404
    r = await client.patch("/users/bob", json={}, headers=headers)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_admin_cannot_demote_or_disable_self(client, fresh_user_store):
    await persistence.users.create_user(
        username="admin", hashed_password=get_password_hash("admin-pw"), role="admin"
    )
    token = await _login(client, "admin", "admin-pw")
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.patch("/users/admin", json={"role": "user"}, headers=headers)
    assert r.status_code == 400
    r = await client.patch("/users/admin", json={"disabled": True}, headers=headers)
    assert r.status_code == 400

    # A no-op self-update (role stays admin) is fine.
    r = await client.patch("/users/admin", json={"role": "admin"}, headers=headers)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_user_create_and_delete_manage_workspace(
    client, fresh_user_store, tmp_path, monkeypatch
):
    """Register creates .workspace/<user> (git-tracked); admin delete purges it."""
    from app.services.workspace import workspace_dir, workspace_root

    monkeypatch.setattr(config.settings, "workspace_root", str(tmp_path / "workspace"))
    await client.post("/register", json={"username": "alice", "password": TEST_PASSWORD})
    ws = workspace_dir("alice")
    assert ws.is_dir()
    assert (ws / ".gitkeep").exists()

    # The workspace root is its own git repo, tracking the new user.
    import subprocess

    out = await asyncio.to_thread(
        subprocess.run,
        ["git", "-C", str(workspace_root()), "log", "--oneline"],
        capture_output=True,
        text=True,
    )
    assert "create workspace for user alice" in out.stdout

    # Admin deletes the user -> dir + git history entry removed.
    await persistence.users.create_user(
        username="admin", hashed_password=get_password_hash("admin-pw"), role="admin"
    )
    token = await _login(client, "admin", "admin-pw")
    r = await client.delete("/users/alice", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 204, r.text
    assert not ws.exists()


@pytest.mark.asyncio
async def test_agent_routes_require_admin_role():
    """Skills/tools CRUD is admin-only; listing is readable by any user."""
    from app.main import create_app as _create_app

    config.settings.database_uri = None
    await persistence.start()
    try:
        await persistence.users.create_user(username="tester", hashed_password="x", role="user")
        token = create_access_token(data={"sub": "tester"})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_create_app()), base_url="http://test"
        ) as http:
            headers = {"Authorization": f"Bearer {token}"}
            # Read-only listing/lookup is open to any authenticated user.
            r = await http.get("/agent/skills", headers=headers)
            assert r.status_code == 200
            # Mutations stay admin-only.
            r = await http.post(
                "/agent/skills",
                headers=headers,
                json={"name": "x", "description": "d", "content": "c"},
            )
            assert r.status_code == 403
            # No token at all -> 401.
            r = await http.get("/agent/skills")
            assert r.status_code == 401
    finally:
        await persistence.stop()
