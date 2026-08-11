"""CRUD for agent resources (skills, MCP tool servers) in the durable store.

Everything is persisted in the LangGraph store — Postgres-backed in production
(`DATABASE_URI` set), in-memory in dev — under well-known namespaces:

- `("agent", "skills")`      — key `/<name>/SKILL.md` = raw markdown file,
  plus optional bundled files at `/<name>/<relative path>` (scripts/,
  references/, assets/, ... — the skill-creator layout), all in the same
  `{"content": ..., "encoding": "utf-8"}` shape StoreBackend uses, so the
  agent's SkillsMiddleware reads exactly what the API writes and its
  filesystem tools (ls/read/write/execute) see the whole skill folder.
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
from ..schema.agent_schema import (
    SkillFileIn,
    SkillFileOut,
    SkillIn,
    SkillOut,
    ToolServerIn,
    ToolServerOut,
)

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_FILE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
"""Skill name + relative file path patterns (schema-aligned, endpoint-side validation)."""

SKILLS_NS = ("agent", "skills")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _skill_file_key(name: str) -> str:
    return f"/{name}/SKILL.md"


def _skill_aux_key(name: str, path: str) -> str:
    return f"/{name}/{path}"


def _skill_path(name: str) -> str:
    return f"{SKILLS_SOURCE}{name}/SKILL.md"


def _skill_value(content: str) -> dict:
    return {
        "content": content,
        "encoding": "utf-8",
        "created_at": _now_iso(),
        "modified_at": _now_iso(),
    }


def _skill_out(name: str, content: str, files: list[SkillFileOut]) -> SkillOut:
    return SkillOut(name=name, content=content, path=_skill_path(name), files=files)


def _skill_markdown(skill: SkillIn) -> str:
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {skill.description}\n"
        "---\n\n"
        f"{skill.content.rstrip()}\n"
    )


def _validate_skill_files(files: list[SkillFileIn]) -> None:
    """Reject reserved paths; schema pattern already covers the rest."""
    for f in files:
        if f.path.lower() == "skill.md":
            raise ValueError("'SKILL.md' is reserved for the skill definition")


async def _skill_files(store: BaseStore, name: str) -> list[SkillFileOut]:
    """All bundled files of a skill (path + content), sorted by path."""
    prefix = f"/{name}/"
    skill_md = f"/{name}/SKILL.md"
    files: list[SkillFileOut] = []
    for item in await store.asearch(SKILLS_NS):
        key = item.key or ""
        if not key.startswith(prefix) or key == skill_md:
            continue
        files.append(
            SkillFileOut(path=key[len(prefix) :], content=(item.value or {}).get("content", ""))
        )
    files.sort(key=lambda f: f.path)
    return files


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


async def list_skills(store: BaseStore) -> list[SkillOut]:
    items = await store.asearch(SKILLS_NS)
    by_name: dict[str, list] = {}
    for item in items:
        m = re.fullmatch(r"/([^/]+)/(.+)", item.key or "")
        if m:
            by_name.setdefault(m.group(1), []).append(item)
    skills: list[SkillOut] = []
    for name, its in by_name.items():
        md = next((it for it in its if it.key == _skill_file_key(name)), None)
        if md is None:
            continue
        files = [
            SkillFileOut(
                path=it.key[len(f"/{name}/") :], content=(it.value or {}).get("content", "")
            )
            for it in its
            if it.key != _skill_file_key(name)
        ]
        files.sort(key=lambda f: f.path)
        skills.append(_skill_out(name, (md.value or {}).get("content", ""), files))
    skills.sort(key=lambda s: s.name)
    return skills


async def get_skill(store: BaseStore, name: str) -> SkillOut | None:
    item = await store.aget(SKILLS_NS, _skill_file_key(name))
    if item is None:
        return None
    return _skill_out(name, (item.value or {}).get("content", ""), await _skill_files(store, name))


async def create_skill(store: BaseStore, skill: SkillIn) -> SkillOut:
    _validate_skill_files(skill.files)
    await store.aput(SKILLS_NS, _skill_file_key(skill.name), _skill_value(_skill_markdown(skill)))
    for f in skill.files:
        await store.aput(SKILLS_NS, _skill_aux_key(skill.name, f.path), _skill_value(f.content))
    files = sorted(
        (SkillFileOut(path=f.path, content=f.content) for f in skill.files), key=lambda f: f.path
    )
    return _skill_out(skill.name, _skill_markdown(skill), files)


async def update_skill(store: BaseStore, name: str, skill: SkillIn) -> SkillOut:
    existing = await store.aget(SKILLS_NS, _skill_file_key(name))
    if existing is None:
        raise KeyError(name)
    _validate_skill_files(skill.files)
    # SKILL.md + listed files are replaced; unlisted bundled files are kept.
    await store.aput(SKILLS_NS, _skill_file_key(name), _skill_value(_skill_markdown(skill)))
    for f in skill.files:
        await store.aput(SKILLS_NS, _skill_aux_key(name, f.path), _skill_value(f.content))
    return _skill_out(skill.name, _skill_markdown(skill), await _skill_files(store, name))


async def delete_skill(store: BaseStore, name: str) -> bool:
    """Delete SKILL.md and every bundled file of the skill."""
    prefix = f"/{name}/"
    items = [it for it in await store.asearch(SKILLS_NS) if (it.key or "").startswith(prefix)]
    if not items:
        return False
    for it in items:
        await store.adelete(SKILLS_NS, it.key)
    return True


async def delete_skill_file(store: BaseStore, name: str, path: str) -> bool:
    """Delete a single bundled file (never SKILL.md)."""
    key = _skill_aux_key(name, path)
    item = await store.aget(SKILLS_NS, key)
    if item is None:
        return False
    await store.adelete(SKILLS_NS, key)
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
    "SKILL_FILE_PATH_RE",
    "SKILL_NAME_RE",
    "create_skill",
    "create_tool_server",
    "delete_skill",
    "delete_skill_file",
    "delete_tool_server",
    "get_skill",
    "get_tool_server",
    "list_skills",
    "list_tool_servers",
    "load_tool_server_configs",
    "update_skill",
    "update_tool_server",
]
