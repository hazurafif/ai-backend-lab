"""Offline tests for the auth routes: /register, /login, /users/me."""

from __future__ import annotations

import httpx
import pytest

from app.core.fake_users import fake_users_db
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
    """Snapshot and restore the demo user store around each test."""
    snapshot = dict(fake_users_db)
    fake_users_db.clear()
    fake_users_db.update(snapshot)
    yield fake_users_db
    fake_users_db.clear()
    fake_users_db.update(snapshot)


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
