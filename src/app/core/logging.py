"""Structured console logging with request-scoped context.

Before this module the app logged via stdlib `logging.getLogger(__name__)`
(~17 modules) but nothing configured it — the root logger defaults to
WARNING, so every `logger.info(...)` call was silently dropped. This module:

- gives the root logger a console handler with a grep-friendly `key=value`
  formatter (level driven by `LOG_LEVEL`, default INFO);
- stamps every log record with the active request's `request_id` /
  `user_id` (contextvars set by `RequestLogMiddleware`), so a tool call deep
  inside an agent run can be traced back to the user's request;
- logs one line per HTTP request (method, path, status, duration, user) via
  `RequestLogMiddleware`, replacing uvicorn's access log (which lacks user
  attribution and per-request timing).

Telemetry beyond logs (OTel/tracing, metrics) is deliberately out of scope:
this service is self-contained and nothing in the compose stack consumes a
trace/metrics backend. The `request_id` plumbing here is exactly what a
future OTel bridge would key on.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .config import settings
from .security import decode_access_token

REQUEST_ID_HEADER = "X-Request-ID"

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


def get_request_id() -> str | None:
    """The active request's id (None outside a request)."""
    return request_id_var.get()


def get_user_id() -> str | None:
    """The active request's authenticated user (None without a bearer token)."""
    return user_id_var.get()


class RequestContextFilter(logging.Filter):
    """Stamp every record with the active request's id and user.

    Runs on the console handler before formatting; absent values become "-".
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.user_id = user_id_var.get() or "-"
        return True


class KeyValueFormatter(logging.Formatter):
    """`time level logger message request_id=.. user=..` — human + grep friendly."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = [
            f"{k}={getattr(record, k)}"
            for k in ("request_id", "user_id")
            if getattr(record, k, None)
        ]
        return " ".join([line, *extras]) if extras else line


_console_handlers: list[logging.Handler] = []
_configured = False


def setup_logging() -> None:
    """Configure the root logger once (idempotent, non-destructive).

    Adds our console handler but never removes handlers it did not create
    (pytest's caplog handler survives). Repeated calls swap only our own
    handler, so `create_app()` racing in tests is safe. Uvicorn's access log
    is silenced — `RequestLogMiddleware` is its replacement.
    """
    global _configured
    root = logging.getLogger()
    for handler in _console_handlers:
        root.removeHandler(handler)
    _console_handlers.clear()

    level = getattr(logging, str(settings.log_level).upper(), None)
    if not isinstance(level, int):
        level = logging.INFO
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    handler.addFilter(RequestContextFilter())
    root.addHandler(handler)
    _console_handlers.append(handler)

    access = logging.getLogger("uvicorn.access")
    access.setLevel(logging.WARNING)
    access.propagate = False
    # Library chatter at INFO (every outbound HTTP call) — the app's own
    # request log and the SSE tool events already cover this path.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


class RequestLogMiddleware(BaseHTTPMiddleware):
    """One log line per HTTP request + request_id/user_id contextvars.

    Generates the request id (honoring an incoming `X-Request-ID`) and echoes
    it back so clients can correlate app logs with their call. The user is
    extracted best-effort from the bearer token — the middleware runs before
    auth dependencies, so a missing/expired token just logs `user=-`.
    Duration is wall-clock until the response starts (time-to-first-byte);
    for SSE streams that is connection setup, not stream lifetime.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex[:16]
        request_id_var.set(rid)
        user_id_var.set(None)
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token_data = decode_access_token(auth[7:].strip())
            if token_data is not None:
                user_id_var.set(token_data.username)

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "%s %s -> %d (%d ms)", request.method, request.url.path, status, round(duration_ms)
            )
        response.headers[REQUEST_ID_HEADER] = rid
        return response


logger = logging.getLogger("app.http")
