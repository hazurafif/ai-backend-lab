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
    "llm_model_kwargs",
    "mask_token",
    "refresh_resolved_connections",
    "resolved_connection",
    "resolved_embeddings",
    "resolved_llm",
    "to_out",
]
