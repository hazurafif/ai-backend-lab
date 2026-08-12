"""Agent resource routes: skills + MCP tool server CRUD, persisted in the store.

Skills apply on the next agent run (SkillsMiddleware reads the backend per
run); tool server changes apply on restart or POST /agent/tools/reconnect.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ....core.database import persistence
from ....core.dependencies import get_admin_user, get_current_user
from ....core.exceptions import Conflict, NotFound
from ....schema.agent_schema import SkillIn, SkillOut, ToolServerIn, ToolServerOut
from ....services import resources
from ....services.agent import AgentRegistry
from ....services.mcp import mcp_servers

router = APIRouter(prefix="/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


@router.get("/skills", response_model=list[SkillOut])
async def list_skills(current_user: dict = Depends(get_current_user)):
    """Read-only listing for any authenticated user (pick global skills for agent configs)."""
    return await resources.list_skills(persistence.store)


@router.post("/skills", response_model=SkillOut, status_code=201)
async def create_skill(body: SkillIn, request: Request, _: dict = Depends(get_admin_user)):
    if await resources.get_skill(persistence.store, body.name):
        raise Conflict(f"Skill '{body.name}' already exists")
    out = await resources.create_skill(persistence.store, body)
    request.app.state.agents.invalidate()
    return out


@router.get("/skills/{name}", response_model=SkillOut)
async def get_skill(name: str, current_user: dict = Depends(get_current_user)):
    """Read-only lookup for any authenticated user."""
    skill = await resources.get_skill(persistence.store, name)
    if skill is None:
        raise NotFound("Skill not found")
    return skill


@router.put("/skills/{name}", response_model=SkillOut)
async def update_skill(
    name: str, body: SkillIn, request: Request, _: dict = Depends(get_admin_user)
):
    try:
        out = await resources.update_skill(persistence.store, name, body)
    except KeyError:
        raise NotFound("Skill not found") from None
    request.app.state.agents.invalidate()
    return out


@router.delete("/skills/{name}", status_code=204)
async def delete_skill(name: str, request: Request, _: dict = Depends(get_admin_user)):
    if not await resources.delete_skill(persistence.store, name):
        raise NotFound("Skill not found")
    request.app.state.agents.invalidate()


@router.delete("/skills/{name}/files/{file_path:path}", status_code=204)
async def delete_skill_file(
    name: str, file_path: str, request: Request, _: dict = Depends(get_admin_user)
):
    """Delete one bundled skill file (scripts/, references/, assets/, ...)."""
    if not resources.SKILL_FILE_PATH_RE.fullmatch(file_path) or file_path.lower() == "skill.md":
        raise HTTPException(status_code=422, detail="Invalid skill file path")
    if not await resources.delete_skill_file(persistence.store, name, file_path):
        raise NotFound("Skill file not found")
    request.app.state.agents.invalidate()


# ---------------------------------------------------------------------------
# MCP tool servers
# ---------------------------------------------------------------------------


@router.get("/tools", response_model=list[ToolServerOut])
async def list_tool_servers(_: dict = Depends(get_admin_user)):
    return await resources.list_tool_servers(persistence.store)


@router.post("/tools", response_model=ToolServerOut, status_code=201)
async def create_tool_server(
    body: ToolServerIn, request: Request, _: dict = Depends(get_admin_user)
):
    if await resources.get_tool_server(persistence.store, body.name):
        raise Conflict(f"Tool server '{body.name}' already exists")
    out = await resources.create_tool_server(persistence.store, body)
    request.app.state.agents.invalidate()
    return out


@router.get("/tools/{name}", response_model=ToolServerOut)
async def get_tool_server(name: str, _: dict = Depends(get_admin_user)):
    server = await resources.get_tool_server(persistence.store, name)
    if server is None:
        raise NotFound("Tool server not found")
    return server


@router.put("/tools/{name}", response_model=ToolServerOut)
async def update_tool_server(
    name: str, body: ToolServerIn, request: Request, _: dict = Depends(get_admin_user)
):
    try:
        out = await resources.update_tool_server(persistence.store, name, body)
    except KeyError:
        raise NotFound("Tool server not found") from None
    request.app.state.agents.invalidate()
    return out


@router.delete("/tools/{name}", status_code=204)
async def delete_tool_server(name: str, request: Request, _: dict = Depends(get_admin_user)):
    if not await resources.delete_tool_server(persistence.store, name):
        raise NotFound("Tool server not found")
    request.app.state.agents.invalidate()


@router.post("/tools/reconnect")
async def reconnect_tools(request: Request, _: dict = Depends(get_admin_user)):
    """Reconnect MCP servers from the store and rebuild the agent (live)."""
    await mcp_servers.connect(store=persistence.store)
    registry: AgentRegistry = request.app.state.agents
    registry.update_mcp_tools(mcp_servers.tools, mcp_servers.tools_by_server)
    request.app.state.agent = await registry.resolve("default", "anonymous")
    request.app.state.backend = registry.backend
    return {"connected": mcp_servers.names, "tools": len(mcp_servers.tools)}
