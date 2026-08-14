"""Connection routes: provider credentials (base URL + API token) CRUD.

Admin-only (connections hold secrets): create/list/detail/update/delete saved
connections at /connections. Mutations refresh the resolved-connection cache
(agent LLM + KB embeddings pick up new credentials on the next agent build)
and drop cached agent graphs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ....core.database import persistence
from ....core.dependencies import get_admin_user
from ....core.exceptions import Conflict, NotFound
from ....schema.connection_schema import ConnectionIn, ConnectionOut
from ....services import connections as connection_service
from ....services.kb.vectorstore import reset_vector_store

router = APIRouter(prefix="/connections", tags=["connections"])


async def _mutated(request: Request, kind: str) -> None:
    """Refresh the resolved-connection cache after a mutation and drop agent graphs."""
    await connection_service.refresh_resolved_connections()
    request.app.state.agents.invalidate()
    if kind in ("embeddings", "weaviate"):
        reset_vector_store()  # next KB build reads the new connection


@router.get("", response_model=list[ConnectionOut])
async def list_connections(_: dict = Depends(get_admin_user)):
    """All saved connections (tokens masked), newest first."""
    return [connection_service.to_out(row) for row in await persistence.connections.list()]


@router.post("", response_model=ConnectionOut, status_code=201)
async def create_connection(
    body: ConnectionIn, request: Request, _: dict = Depends(get_admin_user)
):
    row = await persistence.connections.create(body.model_dump())
    if row is None:
        raise Conflict(f"Connection '{body.name}' already exists")
    await _mutated(request, body.kind)
    return connection_service.to_out(row)


@router.get("/{name}", response_model=ConnectionOut)
async def get_connection(name: str, _: dict = Depends(get_admin_user)):
    row = await persistence.connections.get(name)
    if row is None:
        raise NotFound("Connection not found")
    return connection_service.to_out(row)


@router.put("/{name}", response_model=ConnectionOut)
async def update_connection(
    name: str, body: ConnectionIn, request: Request, _: dict = Depends(get_admin_user)
):
    """Full replace; `api_token` omitted keeps the stored token (write-only)."""
    existing = await persistence.connections.get(name)
    if existing is None:
        raise NotFound("Connection not found")
    patch = body.model_dump()
    if not patch.get("api_token"):
        patch["api_token"] = existing.get("api_token")
    row = await persistence.connections.update(name, patch)
    await _mutated(request, patch["kind"])
    return connection_service.to_out(row)


@router.delete("/{name}", status_code=204)
async def delete_connection(name: str, request: Request, _: dict = Depends(get_admin_user)):
    row = await persistence.connections.get(name)
    if row is None:
        raise NotFound("Connection not found")
    await persistence.connections.delete(name)
    await _mutated(request, row.get("kind", "llm"))
