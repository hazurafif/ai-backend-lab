"""Agent resource routes: skills + MCP tool server CRUD, persisted in the store.

Everything is per-user: skills live under the caller's own skills namespace
and MCP tool servers under the caller's own MCP namespace (see
`services/resources`). The legacy /agent/tools paths are kept for backwards
compatibility — they now address the caller's own servers; the canonical
per-user CRUD is /mcp/servers.

Skills apply on the next agent run (SkillsMiddleware reads the backend per
run); tool server changes apply on the next connect (lazy per user) or
POST /agent/tools/reconnect.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ....core.constants import user_skills_ns
from ....core.database import persistence
from ....core.dependencies import get_admin_user, get_current_user
from ....core.exceptions import Conflict, NotFound
from ....schema.agent_schema import SkillIn, SkillOut
from ....services import resources
from ....services.mcp import mcp_servers

router = APIRouter(prefix="/agent", tags=["agent"])


def _skill_owner(current_user: dict, username: str | None) -> str:
    """Target user for a skill operation: caller by default; admins may pass
    ?username= to manage another user's skills."""
    me = current_user.get("username", "admin")
    if username is None or username == me:
        return me
    if current_user.get("role") != "admin":
        from ....core.exceptions import PermissionDenied

        raise PermissionDenied(detail="Only admins can manage other users' skills")
    return username


# ---------------------------------------------------------------------------
# skills (admin-managed, per-user scoped)
# ---------------------------------------------------------------------------


@router.get("/skills", response_model=list[SkillOut])
async def list_skills(
    request: Request,
    current_user: dict = Depends(get_current_user),
    username: str | None = Query(default=None),
):
    """List a user's skills (default: the caller's own; other users: admin)."""
    target = _skill_owner(current_user, username)
    return await resources.list_skills(persistence.store, user_skills_ns(target))


@router.post("/skills", response_model=SkillOut, status_code=201)
async def create_skill(
    body: SkillIn,
    request: Request,
    admin_user: dict = Depends(get_admin_user),
    username: str | None = Query(default=None),
):
    target = _skill_owner(admin_user, username)
    ns = user_skills_ns(target)
    if await resources.get_skill(persistence.store, body.name, ns):
        raise Conflict(f"Skill '{body.name}' already exists")
    out = await resources.create_skill(persistence.store, body, ns)
    request.app.state.agents.invalidate()
    return out


@router.get("/skills/{name}", response_model=SkillOut)
async def get_skill(
    name: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    username: str | None = Query(default=None),
):
    """Read-only lookup (default: caller's own skills; other users: admin)."""
    target = _skill_owner(current_user, username)
    skill = await resources.get_skill(persistence.store, name, user_skills_ns(target))
    if skill is None:
        raise NotFound("Skill not found")
    return skill


@router.put("/skills/{name}", response_model=SkillOut)
async def update_skill(
    name: str,
    body: SkillIn,
    request: Request,
    admin_user: dict = Depends(get_admin_user),
    username: str | None = Query(default=None),
):
    target = _skill_owner(admin_user, username)
    try:
        out = await resources.update_skill(persistence.store, name, body, user_skills_ns(target))
    except KeyError:
        raise NotFound("Skill not found") from None
    request.app.state.agents.invalidate()
    return out


@router.delete("/skills/{name}", status_code=204)
async def delete_skill(
    name: str,
    request: Request,
    admin_user: dict = Depends(get_admin_user),
    username: str | None = Query(default=None),
):
    target = _skill_owner(admin_user, username)
    if not await resources.delete_skill(persistence.store, name, user_skills_ns(target)):
        raise NotFound("Skill not found")
    request.app.state.agents.invalidate()


@router.delete("/skills/{name}/files/{file_path:path}", status_code=204)
async def delete_skill_file(
    name: str,
    file_path: str,
    request: Request,
    admin_user: dict = Depends(get_admin_user),
    username: str | None = Query(default=None),
):
    """Delete one bundled skill file (scripts/, references/, assets/, ...)."""
    if not resources.SKILL_FILE_PATH_RE.fullmatch(file_path) or file_path.lower() == "skill.md":
        raise HTTPException(status_code=422, detail="Invalid skill file path")
    target = _skill_owner(admin_user, username)
    if not await resources.delete_skill_file(
        persistence.store, name, file_path, user_skills_ns(target)
    ):
        raise NotFound("Skill file not found")
    request.app.state.agents.invalidate()


# ---------------------------------------------------------------------------
# MCP tool servers (per-user: the caller's own servers; canonical CRUD at
# /mcp/servers — these legacy paths keep working, scoped to the caller)
# ---------------------------------------------------------------------------


@router.get("/tools", response_model=list)
async def list_tool_servers(current_user: dict = Depends(get_current_user)):
    return await resources.list_tool_servers(persistence.store, current_user["username"])


@router.post("/tools", response_model=dict, status_code=201)
async def create_tool_server(
    body: dict, request: Request, current_user: dict = Depends(get_current_user)
):
    from pydantic import ValidationError

    from ....schema.agent_schema import ToolServerIn

    try:
        server = ToolServerIn.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from None
    if await resources.get_tool_server(persistence.store, current_user["username"], server.name):
        raise Conflict(f"Tool server '{server.name}' already exists")
    out = await resources.create_tool_server(persistence.store, current_user["username"], server)
    await mcp_servers.invalidate(current_user["username"])
    request.app.state.agents.invalidate()
    return out.model_dump()


@router.get("/tools/{name}", response_model=dict)
async def get_tool_server(name: str, current_user: dict = Depends(get_current_user)):
    server = await resources.get_tool_server(persistence.store, current_user["username"], name)
    if server is None:
        raise NotFound("Tool server not found")
    return server.model_dump()


@router.put("/tools/{name}", response_model=dict)
async def update_tool_server(
    name: str, body: dict, request: Request, current_user: dict = Depends(get_current_user)
):
    from pydantic import ValidationError

    from ....schema.agent_schema import ToolServerIn

    try:
        server = ToolServerIn.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from None
    try:
        out = await resources.update_tool_server(
            persistence.store, current_user["username"], name, server
        )
    except KeyError:
        raise NotFound("Tool server not found") from None
    await mcp_servers.invalidate(current_user["username"])
    request.app.state.agents.invalidate()
    return out.model_dump()


@router.delete("/tools/{name}", status_code=204)
async def delete_tool_server(
    name: str, request: Request, current_user: dict = Depends(get_current_user)
):
    if not await resources.delete_tool_server(persistence.store, current_user["username"], name):
        raise NotFound("Tool server not found")
    await mcp_servers.invalidate(current_user["username"])
    request.app.state.agents.invalidate()


@router.post("/tools/reconnect")
async def reconnect_tools(request: Request, current_user: dict = Depends(get_current_user)):
    """Reconnect the caller's MCP servers from their stored config (live).

    Unreachable servers are recorded per-server (not fatal) so the healthy
    ones still load; the response reports them under `failed`. Cached agent
    graphs are dropped so the next run picks up the new tool set.
    """
    instance = await mcp_servers.connect(current_user["username"], persistence.store)
    request.app.state.agents.invalidate()
    return {
        "connected": [n for n in instance.names if n not in instance.failed],
        "tools": len(instance.tools),
        "failed": instance.failed,
    }
