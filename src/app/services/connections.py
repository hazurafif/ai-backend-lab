"""CRUD for API connections (LLM provider base URL + API key) in the durable store.

A connection is a named `base_url` + `api_key` pair for an OpenAI-compatible
endpoint. It is persisted in the LangGraph store — Postgres-backed in
production (`DATABASE_URI` set), in-memory in dev — under the well-known
namespace `("agent", "connections")` (key = connection name), exactly like
MCP tool servers.

The agent's chat model is built from a stored connection instead of `.env`
keys (`OPENAI_API_KEY` / `OPENAI_BASE_URL`): named agent configs reference a
connection by name, and the connection named `default` (if present) is used
by the builtin `default` agent. The API key is stored as-is (like MCP server
headers/env) and never returned by the API — `ConnectionOut` carries
`has_api_key` only. Agent graphs are cached by spec fingerprint, so any
connection mutation must invalidate the registry (the endpoints do) to pick
up new credentials on the next run.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore

from ..core.constants import CONNECTIONS_NS, DEFAULT_CONNECTION_NAME
from ..schema.connection_schema import ConnectionIn, ConnectionOut, ConnectionUpdate

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _value(conn: ConnectionIn, *, created_at: str | None = None) -> dict[str, Any]:
    updated = _now_iso()
    return {
        "name": conn.name,
        "base_url": conn.base_url,
        "api_key": conn.api_key,
        "created_at": created_at or updated,
        "updated_at": updated,
    }


def _to_out(name: str, value: dict[str, Any]) -> ConnectionOut:
    return ConnectionOut(
        name=name,
        base_url=value["base_url"],
        has_api_key=bool(value.get("api_key")),
        created_at=value.get("created_at") or "",
        updated_at=value.get("updated_at") or "",
    )


async def list_connections(store: BaseStore) -> list[ConnectionOut]:
    items = await store.asearch(CONNECTIONS_NS)
    out = [_to_out(it.key, it.value) for it in items if it.value]
    out.sort(key=lambda c: c.name)
    return out


async def get_connection(store: BaseStore, name: str) -> ConnectionOut | None:
    item = await store.aget(CONNECTIONS_NS, name)
    if item is None:
        return None
    return _to_out(name, item.value)


async def create_connection(store: BaseStore, conn: ConnectionIn) -> ConnectionOut:
    value = _value(conn)
    await store.aput(CONNECTIONS_NS, conn.name, value)
    return _to_out(conn.name, value)


async def update_connection(store: BaseStore, name: str, update: ConnectionUpdate) -> ConnectionOut:
    """Merge the update into the stored value; raises KeyError when unknown.

    Omitted fields (base_url / api_key) keep their stored values, so the API
    key never needs to be re-sent to change the endpoint.
    """
    item = await store.aget(CONNECTIONS_NS, name)
    if item is None:
        raise KeyError(name)
    value = dict(item.value)
    if update.base_url is not None:
        value["base_url"] = update.base_url
    if update.api_key is not None:
        value["api_key"] = update.api_key
    value["updated_at"] = _now_iso()
    await store.aput(CONNECTIONS_NS, name, value)
    return _to_out(name, value)


async def delete_connection(store: BaseStore, name: str) -> bool:
    item = await store.aget(CONNECTIONS_NS, name)
    if item is None:
        return False
    await store.adelete(CONNECTIONS_NS, name)
    return True


async def load_connection(store: BaseStore, name: str) -> dict[str, str] | None:
    """The secret pair for the agent's model builder, or None when unknown."""
    item = await store.aget(CONNECTIONS_NS, name)
    if item is None or not item.value.get("api_key"):
        return None
    return {"base_url": item.value["base_url"], "api_key": item.value["api_key"]}


async def resolve_default_connection(store: BaseStore) -> str | None:
    """Name of the connection used by the builtin `default` agent.

    The connection named `default` (when present) replaces `.env` keys for
    the builtin agent; None falls back to the env-based behavior.
    """
    item = await store.aget(CONNECTIONS_NS, DEFAULT_CONNECTION_NAME)
    return DEFAULT_CONNECTION_NAME if item is not None and item.value.get("api_key") else None


__all__ = [
    "create_connection",
    "delete_connection",
    "get_connection",
    "list_connections",
    "load_connection",
    "resolve_default_connection",
    "update_connection",
]
