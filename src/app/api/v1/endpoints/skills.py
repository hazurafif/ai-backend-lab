"""User-scoped skills routes ("my skills").

Mirror of the admin-managed global skills (`/agent/skills`) but private to
each user, persisted under ("user", "skills", <username>). User skills can be
attached to the user's own agent configs (see /agents); global agents only
reference global skills. Skills apply on the next agent run — selecting a
skill snapshot-copies it into the agent's namespace at config save time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ....core.constants import user_skills_ns
from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....core.exceptions import Conflict, NotFound
from ....schema.agent_schema import SkillIn, SkillOut
from ....services import resources

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("", response_model=list[SkillOut])
async def list_my_skills(current_user: dict = Depends(get_current_user)):
    return await resources.list_skills(persistence.store, user_skills_ns(current_user["username"]))


@router.post("", response_model=SkillOut, status_code=201)
async def create_my_skill(
    body: SkillIn, request: Request, current_user: dict = Depends(get_current_user)
):
    ns = user_skills_ns(current_user["username"])
    if await resources.get_skill(persistence.store, body.name, ns):
        raise Conflict(f"Skill '{body.name}' already exists")
    out = await resources.create_skill(persistence.store, body, ns)
    request.app.state.agents.invalidate()
    return out


@router.get("/{name}", response_model=SkillOut)
async def get_my_skill(name: str, current_user: dict = Depends(get_current_user)):
    skill = await resources.get_skill(
        persistence.store, name, user_skills_ns(current_user["username"])
    )
    if skill is None:
        raise NotFound("Skill not found")
    return skill


@router.put("/{name}", response_model=SkillOut)
async def update_my_skill(
    name: str, body: SkillIn, request: Request, current_user: dict = Depends(get_current_user)
):
    try:
        out = await resources.update_skill(
            persistence.store, name, body, user_skills_ns(current_user["username"])
        )
    except KeyError:
        raise NotFound("Skill not found") from None
    request.app.state.agents.invalidate()
    return out


@router.delete("/{name}", status_code=204)
async def delete_my_skill(
    name: str, request: Request, current_user: dict = Depends(get_current_user)
):
    if not await resources.delete_skill(
        persistence.store, name, user_skills_ns(current_user["username"])
    ):
        raise NotFound("Skill not found")
    request.app.state.agents.invalidate()
    return None


@router.delete("/{name}/files/{file_path:path}", status_code=204)
async def delete_my_skill_file(
    name: str, file_path: str, request: Request, current_user: dict = Depends(get_current_user)
):
    """Delete one bundled skill file (scripts/, references/, assets/, ...)."""
    if not resources.SKILL_FILE_PATH_RE.fullmatch(file_path) or file_path.lower() == "skill.md":
        raise HTTPException(status_code=422, detail="Invalid skill file path")
    if not await resources.delete_skill_file(
        persistence.store, name, file_path, user_skills_ns(current_user["username"])
    ):
        raise NotFound("Skill file not found")
    request.app.state.agents.invalidate()
    return None
