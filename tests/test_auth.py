"""Offline tests for the auth routes: /register, /login, /users/me.

The users store is dict-backed here (no Postgres in tests), and the default
admin seeding is exercised by `test_default_admin_seeded_on_first_start`.
"""

from __future__ import annotations

import httpx
import pytest

from app.core import config
from app.core.database import persistence
from app.core.security import create_access_token, verify_password
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
            "password": "super-secret",
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
    login = await client.post("/login", data={"username": "alice", "password": "super-secret"})
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_duplicate_username_conflicts(client, fresh_user_store):
    first = await client.post("/register", json={"username": "bob", "password": "super-secret"})
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
        "/register", json={"username": "bad name!", "password": "super-secret"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_me_returns_profile(client, fresh_user_store):
    await client.post(
        "/register",
        json={
            "username": "alice",
            "password": "super-secret",
            "email": "alice@example.com",
            "full_name": "Alice Example",
        },
    )
    login = await client.post("/login", data={"username": "alice", "password": "super-secret"})
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

        # Restarting must not create a second admin.
        await persistence.stop()
        await persistence.start()
        assert await persistence.users.count() == 1

        # The seeded username is reserved by the register API.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            response = await client.post(
                "/register", json={"username": "admin", "password": "super-secret"}
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
    finally:
        await persistence.stop()
