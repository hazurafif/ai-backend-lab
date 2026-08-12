"""Offline tests for the auth hardening batch:

- POST /refresh (refresh token flow, type separation)
- POST /users/me/password (self-service password change)
- admin POST /users + DELETE /users/{username}
- login rate limiting (429 past the cap, success resets)
"""

from __future__ import annotations

import httpx
import pytest
from conftest import TEST_NEW_PASSWORD, TEST_PASSWORD

from app.core.database import persistence
from app.core.rate_limit import login_limiter
from app.core.security import get_password_hash
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


async def _login(client, username: str, password: str) -> dict:
    r = await client.post("/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# refresh tokens
# ---------------------------------------------------------------------------


async def test_refresh_token_flow(client, fresh_user_store):
    await client.post("/register", json={"username": "alice", "password": TEST_PASSWORD})
    tokens = await _login(client, "alice", TEST_PASSWORD)
    assert tokens["refresh_token"], "login must return a refresh token"

    # Refresh -> new access token that works.
    r = await client.post("/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert "refresh_token" not in body, "refresh response only returns an access token"

    me = await client.get("/users/me/", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["username"] == "alice"

    # An access token is not a valid refresh token; garbage is rejected too.
    r = await client.post("/refresh", json={"refresh_token": tokens["access_token"]})
    assert r.status_code == 401, r.text
    r = await client.post("/refresh", json={"refresh_token": "not-a-token"})
    assert r.status_code == 401, r.text


async def test_refresh_rejects_disabled_user(client, fresh_user_store):
    await client.post("/register", json={"username": "alice", "password": TEST_PASSWORD})
    tokens = await _login(client, "alice", TEST_PASSWORD)
    await persistence.users.update_user("alice", disabled=True)

    r = await client.post("/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 403, r.text


# ---------------------------------------------------------------------------
# self-service password change
# ---------------------------------------------------------------------------


async def test_change_own_password(client, fresh_user_store):
    await client.post("/register", json={"username": "alice", "password": TEST_PASSWORD})
    token = (await _login(client, "alice", TEST_PASSWORD))["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Wrong old password -> 400.
    r = await client.post(
        "/users/me/password",
        json={"old_password": "wrong", "new_password": TEST_NEW_PASSWORD},
        headers=headers,
    )
    assert r.status_code == 400, r.text

    r = await client.post(
        "/users/me/password",
        json={"old_password": TEST_PASSWORD, "new_password": TEST_NEW_PASSWORD},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["username"] == "alice"

    # Old password no longer works; the new one does.
    r = await client.post("/login", data={"username": "alice", "password": TEST_PASSWORD})
    assert r.status_code == 401, r.text
    r = await client.post("/login", data={"username": "alice", "password": TEST_NEW_PASSWORD})
    assert r.status_code == 200, r.text

    # Short new password -> 422.
    r = await client.post(
        "/users/me/password",
        json={"old_password": TEST_NEW_PASSWORD, "new_password": "short"},
        headers=headers,
    )
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# admin user creation + deletion
# ---------------------------------------------------------------------------


async def test_admin_create_and_delete_user(client, fresh_user_store):
    await persistence.users.create_user(
        username="admin", hashed_password=get_password_hash("admin-pw"), role="admin"
    )
    admin_token = (await _login(client, "admin", "admin-pw"))["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Admin creates a user, optionally with the admin role.
    r = await client.post(
        "/users",
        json={"username": "bob", "password": TEST_PASSWORD, "role": "admin"},
        headers=admin_headers,
    )
    assert r.status_code == 201, r.text
    assert r.json()["role"] == "admin"
    assert "hashed_password" not in r.json()

    # Duplicate -> 409.
    r = await client.post(
        "/users", json={"username": "bob", "password": TEST_PASSWORD}, headers=admin_headers
    )
    assert r.status_code == 409, r.text

    # Non-admin cannot create users.
    await client.post("/register", json={"username": "carol", "password": TEST_PASSWORD})
    carol_token = (await _login(client, "carol", TEST_PASSWORD))["access_token"]
    r = await client.post(
        "/users",
        json={"username": "dave", "password": TEST_PASSWORD},
        headers={"Authorization": f"Bearer {carol_token}"},
    )
    assert r.status_code == 403, r.text

    # Delete bob; his token stops working.
    bob_token = (await _login(client, "bob", TEST_PASSWORD))["access_token"]
    r = await client.delete("/users/bob", headers=admin_headers)
    assert r.status_code == 204, r.text
    r = await client.get("/users/me/", headers={"Authorization": f"Bearer {bob_token}"})
    assert r.status_code == 401, r.text

    # Lockout guards and unknown user.
    r = await client.delete("/users/admin", headers=admin_headers)
    assert r.status_code == 400, r.text
    r = await client.delete("/users/ghost", headers=admin_headers)
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# login rate limiting
# ---------------------------------------------------------------------------


async def test_login_rate_limit(client, fresh_user_store):
    login_limiter.clear()
    await client.post("/register", json={"username": "ratelimit", "password": TEST_PASSWORD})

    # A single failure is allowed; a successful login resets the counter.
    r = await client.post("/login", data={"username": "ratelimit", "password": "wrong"})
    assert r.status_code == 401
    r = await client.post("/login", data={"username": "ratelimit", "password": TEST_PASSWORD})
    assert r.status_code == 200

    # Hammer with failures: the cap (10) is enforced, then 429.
    for i in range(10):
        r = await client.post("/login", data={"username": "ratelimit", "password": "wrong"})
        assert r.status_code == 401, f"attempt {i} should be 401"
    r = await client.post("/login", data={"username": "ratelimit", "password": TEST_PASSWORD})
    assert r.status_code == 429, r.text

    # Other usernames are unaffected (key includes the username).
    r = await client.post("/login", data={"username": "nobody", "password": "x"})
    assert r.status_code == 401

    login_limiter.clear()
