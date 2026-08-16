"""Agent config routes: CRUD for customizable agent profiles.

Agents bundle model + system prompt + skill/tool selection into a named
profile referenced by the `agent` field of /chat and /api/chat. Users manage
their own agents; `scope: "global"` agents (shared by all users) are
admin-only. The built-in `default` agent is read-only.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....core.exceptions import BadRequest, Conflict, NotFound, PermissionDenied
from ....schema.agent_config_schema import AgentConfigIn, AgentConfigOut
from ....services import agent_configs, resources

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentConfigOut])
async def list_agents(current_user: dict = Depends(get_current_user)):
    """The builtin default, the user's agents, then global agents (by name)."""
    return await agent_configs.list_configs(persistence.store, current_user["username"])


@router.post("", response_model=AgentConfigOut, status_code=201)
async def create_agent(
    body: AgentConfigIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Create an agent config (skills/tools are validated + snapshotted)."""
    if body.scope == "global" and current_user.get("role") != "admin":
        raise PermissionDenied(detail="Admin role required for global agents")
    try:
        out = await agent_configs.create_config(
            persistence.store,
            body,
            current_user["username"],
            known_servers=await resources.stored_tool_server_names(
                persistence.store, current_user["username"]
            ),
        )
    except KeyError as exc:
        raise Conflict(f"Name already taken: {exc.args[0]}") from None
    except ValueError as exc:
        raise BadRequest(str(exc)) from None
    request.app.state.agents.invalidate()
    return out


@router.get("/{name}", response_model=AgentConfigOut)
async def get_agent(name: str, current_user: dict = Depends(get_current_user)):
    out = await agent_configs.get_config(persistence.store, name, current_user["username"])
    if out is None:
        raise NotFound(f"Agent '{name}' not found")
    return out


@router.post("/{name}/test")
async def test_agent(name: str, request: Request, current_user: dict = Depends(get_current_user)):
    """Dry-run: resolve the config and build the graph (validates model + skills + tools).

    Returns the resolved spec; a 400 means the config cannot be built (e.g.
    an invalid provider:model string). The built graph is cached.
    """
    try:
        graph = await request.app.state.agents.resolve(name, current_user["username"])
    except KeyError:
        raise NotFound(f"Agent '{name}' not found") from None
    except Exception as exc:
        raise BadRequest(f"Agent '{name}' cannot be built: {exc}") from None
    spec = await agent_configs.load_spec(persistence.store, name, current_user["username"])
    return {
        "status": "ok",
        "name": name,
        "graph_built": graph is not None,
        "model": spec.model if spec else None,
        "skills": spec.skills if spec else None,
        "tools": spec.tools if spec else None,
        "temperature": spec.temperature if spec else None,
        "thinking": spec.thinking if spec else None,
    }


@router.put("/{name}", response_model=AgentConfigOut)
async def update_agent(
    name: str,
    body: AgentConfigIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Replace an agent the user owns (admin: also global agents)."""
    if name == agent_configs.DEFAULT_AGENT_NAME:
        raise PermissionDenied(detail="The built-in 'default' agent is read-only")
    if body.scope == "global" and current_user.get("role") != "admin":
        raise PermissionDenied(detail="Admin role required for global agents")
    try:
        out = await agent_configs.update_config(
            persistence.store,
            name,
            body,
            current_user["username"],
            known_servers=await resources.stored_tool_server_names(
                persistence.store, current_user["username"]
            ),
        )
    except KeyError:
        raise NotFound(f"Agent '{name}' not found") from None
    except ValueError as exc:
        raise BadRequest(str(exc)) from None
    request.app.state.agents.invalidate()
    return out


@router.delete("/{name}", status_code=204)
async def delete_agent(name: str, request: Request, current_user: dict = Depends(get_current_user)):
    """Delete the user's agent (or a global one, for admins)."""
    if name == agent_configs.DEFAULT_AGENT_NAME:
        raise PermissionDenied(detail="The built-in 'default' agent is read-only")
    out = await agent_configs.get_config(persistence.store, name, current_user["username"])
    if out is None:
        raise NotFound(f"Agent '{name}' not found")
    if out.scope == "global" and current_user.get("role") != "admin":
        raise PermissionDenied(detail="Admin role required for global agents")
    await agent_configs.delete_config(
        persistence.store, name, current_user["username"], allow_global=out.scope == "global"
    )
    request.app.state.agents.invalidate()
    return None
