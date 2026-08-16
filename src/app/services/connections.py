"""Provider connections: resolved defaults + credential helpers.

Connections (base URL + API token) are managed via the /connections CRUD API
and persisted in `core/database.ConnectionStore`. Consumers (the agent's LLM
model, KB embeddings) never touch .env when a connection exists: the default
connection per kind is resolved into a process-wide cache at startup and
refreshed after every CRUD mutation, and sync code (e.g. `build_embeddings`)
reads the cache directly.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..schema.connection_schema import ConnectionKind

# Kinds resolved into the cache; consumers read these synchronously.
RESOLVED_KINDS: tuple[ConnectionKind, ...] = ("llm", "embeddings")

_resolved: dict[ConnectionKind, dict | None] = {kind: None for kind in RESOLVED_KINDS}


async def refresh_resolved_connections() -> None:
    """Re-resolve the default connection per kind from the store.

    Called at startup (lifespan) and after every /connections mutation, so the
    agent picks up new credentials on the next graph build.
    """
    from ..core.database import persistence

    for kind in RESOLVED_KINDS:
        row = await persistence.connections.get_default(kind)
        _resolved[kind] = row


def resolved_connection(kind: ConnectionKind) -> dict | None:
    """The cached default connection of a kind, or None when none is saved."""
    return _resolved.get(kind)


def resolved_llm() -> dict | None:
    """Cached default `llm` connection (base_url/api_token for the agent model)."""
    return resolved_connection("llm")


def resolved_embeddings() -> dict | None:
    """Cached default `embeddings` connection (base_url/api_token for embeddings)."""
    return resolved_connection("embeddings")


def llm_model_kwargs() -> dict[str, Any]:
    """init_chat_model kwargs from the default llm connection (empty = env only)."""
    conn = resolved_llm()
    if conn is None:
        return {}
    kwargs: dict[str, Any] = {}
    if conn.get("base_url"):
        kwargs["base_url"] = conn["base_url"]
    if conn.get("api_token"):
        kwargs["api_key"] = conn["api_token"]
    return kwargs


def llm_model_name() -> str | None:
    """The model of the default llm connection (`extra.model`), or None.

    The connection's model is the source for the builtin `default` agent
    (no env fallback); an explicit spec/agent model always wins over it.
    """
    conn = resolved_llm()
    if conn is None:
        return None
    extra = conn.get("extra") or {}
    return extra.get("model") or None


async def fetch_models(
    conn: dict, *, request_timeout: float | None = 8.0, client: httpx.AsyncClient | None = None
) -> list[dict]:
    """Model ids of one connection via its OpenAI-compatible `GET {base_url}/models`.

    Raises on missing base_url, non-2xx responses and network errors; callers
    (e.g. `discover_models`) surface per-source failures as `error` entries.
    `client` is a test hook (httpx.MockTransport) — headers still come from
    the connection row.
    """
    base_url = (conn.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("connection has no base_url")
    headers: dict[str, str] = {}
    token = conn.get("api_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=request_timeout, headers=headers)
    try:
        resp = await client.get(f"{base_url}/models")
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owns_client:
            await client.aclose()
    out: list[dict] = []
    for model in data.get("data") or []:
        if not isinstance(model, dict) or not model.get("id"):
            continue
        out.append(
            {"id": model["id"], "created": model.get("created"), "owned_by": model.get("owned_by")}
        )
    return out


async def discover_models(client: httpx.AsyncClient | None = None) -> list[dict]:
    """Aggregate model lists from every saved llm connection (best-effort).

    Queries each `llm` connection's /models endpoint; a failing source is
    reported as `{"error": ...}` instead of failing the whole call, so the
    frontend can render a picker across all configured sources (opencode,
    gemini, openrouter, ...) and show per-source health.
    """
    from ..core.database import persistence

    rows = await persistence.connections.list()
    sources: list[dict] = []
    for row in rows:
        if row.get("kind") != "llm":
            continue
        entry: dict[str, Any] = {
            "connection": row["name"],
            "base_url": row.get("base_url"),
            "is_default": bool(row.get("is_default")),
            "models": [],
            "error": None,
        }
        try:
            entry["models"] = await fetch_models(row, client=client)
        except Exception as exc:
            entry["error"] = str(exc)
        sources.append(entry)
    return sources


def model_kwargs_from(conn: dict) -> dict[str, Any]:
    """init_chat_model kwargs (base_url + api_key) for a specific connection row."""
    kwargs: dict[str, Any] = {}
    if conn.get("base_url"):
        kwargs["base_url"] = conn["base_url"]
    if conn.get("api_token"):
        kwargs["api_key"] = conn["api_token"]
    return kwargs


def mask_token(token: str | None) -> str | None:
    """Mask an api_token for API responses: first 4 + last 4 chars.

    Short tokens (<10 chars) are never echoed — only a marker is returned.
    """
    if not token:
        return None
    if len(token) < 10:
        return "••••"
    return f"{token[:4]}…{token[-4:]}"


def to_out(row: dict) -> dict:
    """Store row -> API response dict (token masked, has_token flag)."""
    token = row.get("api_token")
    out = {k: v for k, v in row.items() if k != "api_token"}
    out["api_token"] = mask_token(token)
    out["has_token"] = bool(token)
    return out


__all__ = [
    "discover_models",
    "fetch_models",
    "llm_model_kwargs",
    "llm_model_name",
    "mask_token",
    "model_kwargs_from",
    "refresh_resolved_connections",
    "resolved_connection",
    "resolved_embeddings",
    "resolved_llm",
    "to_out",
]
