"""API connection routes: LLM provider credentials (base URL + API key).

Connections are persisted in the durable store and used to build the agent's
chat model instead of `.env` keys (`OPENAI_API_KEY` / `OPENAI_BASE_URL`).
Mutations are admin-only; reads are available to any authenticated user so
they can reference connection names in agent configs (`connection` field on
/agents). The connection named `default` is used by the builtin `default`
agent. The API key is write-only — it is never returned, and updates may
omit it to keep the stored key.

Agent graphs are cached, so every mutation invalidates the registry; the
next chat run rebuilds with the new credentials.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ....core.database import persistence
from ....core.dependencies import get_admin_user, get_current_user
from ....core.exceptions import BadRequest, Conflict, NotFound
from ....schema.connection_schema import ConnectionIn, ConnectionOut, ConnectionUpdate
from ....services import connections

router = APIRouter(prefix="/agent/connections", tags=["agent"])


@router.get("", response_model=list[ConnectionOut])
async def list_connections(_: dict = Depends(get_current_user)):
    """All connections (no API keys) — pick a name for an agent config."""
    return await connections.list_connections(persistence.store)


@router.post("", response_model=ConnectionOut, status_code=201)
async def create_connection(
    body: ConnectionIn, request: Request, _: dict = Depends(get_admin_user)
):
    if await connections.get_connection(persistence.store, body.name):
        raise Conflict(f"Connection '{body.name}' already exists")
    out = await connections.create_connection(persistence.store, body)
    request.app.state.agents.invalidate()
    return out


@router.get("/{name}", response_model=ConnectionOut)
async def get_connection(name: str, _: dict = Depends(get_current_user)):
    conn = await connections.get_connection(persistence.store, name)
    if conn is None:
        raise NotFound("Connection not found")
    return conn


@router.put("/{name}", response_model=ConnectionOut)
async def update_connection(
    name: str, body: ConnectionUpdate, request: Request, _: dict = Depends(get_admin_user)
):
    """Merge the update; omitted fields (incl. api_key) keep their stored values."""
    if body.base_url is None and body.api_key is None:
        raise BadRequest(detail="Provide at least one of base_url, api_key")
    try:
        out = await connections.update_connection(persistence.store, name, body)
    except KeyError:
        raise NotFound("Connection not found") from None
    request.app.state.agents.invalidate()
    return out


@router.delete("/{name}", status_code=204)
async def delete_connection(name: str, request: Request, _: dict = Depends(get_admin_user)):
    if not await connections.delete_connection(persistence.store, name):
        raise NotFound("Connection not found")
    request.app.state.agents.invalidate()
