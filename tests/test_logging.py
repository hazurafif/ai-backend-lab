"""Offline tests for structured logging (core/logging.py).

Requests go through `httpx.AsyncClient` + `ASGITransport` on a minimal app
(no auth, no persistence) — validates the request middleware, contextvar
propagation, user attribution, and the log formatter without touching the
real app or any external service.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI

from app.core import logging as app_logging
from app.core.security import create_access_token


def _minimal_app() -> FastAPI:
    """FastAPI app exposing the middleware's contextvars to assert on."""
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        return {
            "request_id": app_logging.get_request_id(),
            "user_id": app_logging.get_user_id(),
        }

    app.add_middleware(app_logging.RequestLogMiddleware)
    return app


@pytest.mark.asyncio
async def test_middleware_echoes_incoming_request_id() -> None:
    app = _minimal_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping", headers={"X-Request-ID": "abc123"})

    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == "abc123"
    # The contextvar set in the middleware is visible inside the endpoint.
    assert resp.json()["request_id"] == "abc123"


@pytest.mark.asyncio
async def test_middleware_generates_request_id_when_absent() -> None:
    app = _minimal_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    rid = resp.headers["x-request-id"]
    assert rid and rid != "-"
    assert resp.json()["request_id"] == rid


@pytest.mark.asyncio
async def test_middleware_attributes_user_from_bearer_token() -> None:
    app = _minimal_app()
    token = create_access_token({"sub": "alice"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json()["user_id"] == "alice"


@pytest.mark.asyncio
async def test_middleware_leaves_user_unset_without_token() -> None:
    app = _minimal_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    assert resp.json()["user_id"] is None


def test_request_context_filter_stamps_record() -> None:
    app_logging.request_id_var.set("rid-1")
    app_logging.user_id_var.set("alice")
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", (), None)

    assert app_logging.RequestContextFilter().filter(record) is True
    assert record.request_id == "rid-1"
    assert record.user_id == "alice"


def test_request_context_filter_defaults_to_dash_outside_request() -> None:
    app_logging.request_id_var.set(None)
    app_logging.user_id_var.set(None)
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", (), None)

    app_logging.RequestContextFilter().filter(record)
    assert record.request_id == "-"
    assert record.user_id == "-"


def test_key_value_formatter_appends_context_extras() -> None:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", (), None)
    record.request_id = "rid-1"
    record.user_id = "alice"
    formatter = app_logging.KeyValueFormatter("%(levelname)s %(message)s")

    line = formatter.format(record)
    assert "INFO hello" in line
    assert line.endswith("request_id=rid-1 user_id=alice")


def test_key_value_formatter_omits_missing_extras() -> None:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", (), None)
    record.request_id = None
    record.user_id = None
    formatter = app_logging.KeyValueFormatter("%(levelname)s %(message)s")

    assert formatter.format(record) == "INFO hello"


def test_setup_logging_is_idempotent_and_non_destructive() -> None:
    # Prior invocations (other tests importing app.main) must not matter:
    # re-running keeps the root logger at the configured level, swaps only
    # our own console handler, and never duplicates it.
    before = list(logging.getLogger().handlers)
    app_logging.setup_logging()
    after = list(logging.getLogger().handlers)

    new_handlers = [h for h in after if h not in before]
    assert len(new_handlers) == 1  # exactly one of our handlers added
    assert logging.getLogger().level == logging.getLevelName("INFO")


def test_setup_logging_silences_uvicorn_access() -> None:
    app_logging.setup_logging()
    access = logging.getLogger("uvicorn.access")
    assert access.level >= logging.WARNING
    assert access.propagate is False
