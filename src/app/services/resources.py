"""CRUD for agent resources (skills, MCP tool servers) in the durable store.

Everything is persisted in the LangGraph store — Postgres-backed in production
(`DATABASE_URI` set), in-memory in dev — under well-known namespaces:

- `("agent", "skills")`      — key `/<name>/SKILL.md`, value = raw markdown file
  (same `{"content": ..., "encoding": "utf-8"}` shape StoreBackend uses, so the
  agent's SkillsMiddleware reads exactly what the API writes).
- `("agent", "mcp_servers")` — key `<name>`, value = MCP server config dict
  (same shape as `mcp_servers.json`; `enabled: false` entries are skipped).

Skills become visible to the agent on the next run (SkillsMiddleware reads the
backend per run — no agent rebuild). MCP server changes apply after a restart
or `POST /agent/tools/reconnect`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from langgraph.store.base import BaseStore

from ..core.config import settings
from ..core.constants import SKILLS_SOURCE, TOOL_SERVERS_NS
from ..schema.agent_schema import SkillIn, SkillOut, ToolServerIn, ToolServerOut

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
"""Agent Skills spec: lowercase alphanumeric + hyphens, no edges/doubles."""

SKILLS_NS = ("agent", "skills")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _skill_file_key(name: str) -> str:
    return f"/{name}/SKILL.md"


def _skill_path(name: str) -> str:
    return f"{SKILLS_SOURCE}{name}/SKILL.md"


def _skill_out(name: str, content: str) -> SkillOut:
    return SkillOut(name=name, content=content, path=_skill_path(name))


def _skill_markdown(skill: SkillIn) -> str:
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {skill.description}\n"
        "---\n\n"
        f"{skill.content.rstrip()}\n"
    )


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


async def list_skills(store: BaseStore) -> list[SkillOut]:
    items = await store.asearch(SKILLS_NS)
    skills: list[SkillOut] = []
    for item in items:
        m = re.fullmatch(r"/([^/]+)/SKILL\.md", item.key or "")
        if not m:
            continue
        content = (item.value or {}).get("content", "")
        skills.append(_skill_out(m.group(1), content))
    skills.sort(key=lambda s: s.name)
    return skills


async def get_skill(store: BaseStore, name: str) -> SkillOut | None:
    item = await store.aget(SKILLS_NS, _skill_file_key(name))
    if item is None:
        return None
    return _skill_out(name, (item.value or {}).get("content", ""))


async def create_skill(store: BaseStore, skill: SkillIn) -> SkillOut:
    await store.aput(
        SKILLS_NS,
        _skill_file_key(skill.name),
        {
            "content": _skill_markdown(skill),
            "encoding": "utf-8",
            "created_at": _now_iso(),
            "modified_at": _now_iso(),
        },
    )
    return _skill_out(skill.name, _skill_markdown(skill))


async def update_skill(store: BaseStore, name: str, skill: SkillIn) -> SkillOut:
    existing = await store.aget(SKILLS_NS, _skill_file_key(name))
    if existing is None:
        raise KeyError(name)
    await create_skill(store, skill)
    return _skill_out(skill.name, _skill_markdown(skill))


async def delete_skill(store: BaseStore, name: str) -> bool:
    item = await store.aget(SKILLS_NS, _skill_file_key(name))
    if item is None:
        return False
    await store.adelete(SKILLS_NS, _skill_file_key(name))
    return True


# ---------------------------------------------------------------------------
# MCP tool servers
# ---------------------------------------------------------------------------


async def list_tool_servers(store: BaseStore) -> list[ToolServerOut]:
    items = await store.asearch(TOOL_SERVERS_NS)
    servers = [ToolServerOut(name=it.key, **it.value) for it in items if it.value]
    servers.sort(key=lambda s: s.name)
    return servers


async def get_tool_server(store: BaseStore, name: str) -> ToolServerOut | None:
    item = await store.aget(TOOL_SERVERS_NS, name)
    if item is None:
        return None
    return ToolServerOut(name=name, **item.value)


async def create_tool_server(store: BaseStore, server: ToolServerIn) -> ToolServerOut:
    await store.aput(TOOL_SERVERS_NS, server.name, server.config())
    return ToolServerOut(name=server.name, **server.config())


async def update_tool_server(store: BaseStore, name: str, server: ToolServerIn) -> ToolServerOut:
    existing = await store.aget(TOOL_SERVERS_NS, name)
    if existing is None:
        raise KeyError(name)
    await store.aput(TOOL_SERVERS_NS, server.name, server.config())
    return ToolServerOut(name=server.name, **server.config())


async def delete_tool_server(store: BaseStore, name: str) -> bool:
    item = await store.aget(TOOL_SERVERS_NS, name)
    if item is None:
        return False
    await store.adelete(TOOL_SERVERS_NS, name)
    return True


async def load_tool_server_configs(store: BaseStore) -> dict:
    """MCP server configs for the agent: store first, env/file fallback."""
    items = await store.asearch(TOOL_SERVERS_NS)
    if items:
        return {it.key: it.value for it in items if it.value and it.value.get("enabled", True)}
    return settings.load_mcp_servers()


# Re-exported so the /skills/ source path used by callers stays in one place.
__all__ = [
    "SKILLS_SOURCE",
    "SKILL_NAME_RE",
    "create_skill",
    "create_tool_server",
    "delete_skill",
    "delete_tool_server",
    "get_skill",
    "get_tool_server",
    "list_skills",
    "list_tool_servers",
    "load_tool_server_configs",
    "update_skill",
    "update_tool_server",
]
