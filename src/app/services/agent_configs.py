"""Agent configs (customizable agent profiles) in the durable store.

An agent config bundles model + system prompt + skill/tool selection into a
named profile that chat requests reference via the `agent` field. Persistence
follows the other resource services: LangGraph store, Postgres-backed in
production, in-memory in dev.

Namespaces:

- `("agent", "agents")`      — global agents (shared by all users, admin-managed)
- `("agents", <username>)`   — a user's own agents
- `("agent", "agent_skills", <owner>, <name>)` — per-agent skill snapshot
  (SKILL.md + bundled files copied from the global skills store when the
  config references them; the agent's SkillsMiddleware reads this namespace
  via a `/skills/<owner>/<name>/` backend route)

The built-in `default` agent is synthesized from `DEEPAGENTS_MODEL` (or the
default `llm` connection's `extra.model` when unset) + `SYSTEM_PROMPT` env
settings and cannot be created or deleted through the API. When neither model
source exists, chats return 503 with setup instructions — the app always
starts and the model is configured in-app later.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore

from ..core.config import settings
from ..core.constants import (
    DEFAULT_AGENT_NAME,
    GLOBAL_AGENTS_NS,
    SKILLS_SOURCE,
    agent_skills_ns,
    agent_skills_source,
    user_agents_ns,
    user_skills_ns,
)
from ..schema.agent_config_schema import AgentConfigIn, AgentConfigOut
from ..services import settings as runtime_settings

logger = logging.getLogger(__name__)

_GLOBAL_OWNER = "global"
_SYSTEM_OWNER = "system"


@dataclass
class AgentSpec:
    """Resolved, build-ready agent definition (what the registry builds)."""

    name: str
    model: str | None  # None = fall back to the default llm connection's extra.model
    system_prompt: str | None
    skills: list[str] | None  # None = inherit global skills, [] = none
    tools: list[str] | None  # None = inherit all tools, [] = none
    temperature: float | None
    interrupt_on: dict[str, bool] | None
    thinking: str | None  # reasoning-effort level: none..minimal..max
    owner: str = _SYSTEM_OWNER
    builtin: bool = False
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    _fingerprint: str = field(default="", init=False, repr=False)

    @property
    def skills_source(self) -> str | None:
        """Filesystem source path for the SkillsMiddleware, or None (no skills)."""
        if self.builtin or self.skills is None:
            return SKILLS_SOURCE
        if not self.skills:
            return None
        return agent_skills_source(self.owner, self.name)

    @property
    def skills_ns(self) -> tuple[str, ...] | None:
        if self.skills_source is None or self.builtin or self.skills is None:
            return None
        return agent_skills_ns(self.owner, self.name)

    def fingerprint(self) -> str:
        """Stable cache key: everything that changes the compiled graph."""
        if not self._fingerprint:
            payload = json.dumps(
                [
                    self.model,
                    self.system_prompt,
                    self.skills,
                    self.tools,
                    self.temperature,
                    self.interrupt_on,
                    self.thinking,
                    self.skills_source,
                ],
                sort_keys=True,
                default=str,
            )
            digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
            self._fingerprint = f"{self.name}:{digest}"
        return self._fingerprint


def default_spec() -> AgentSpec:
    """The built-in agent, seeded from env settings (never stored)."""
    return AgentSpec(
        name=DEFAULT_AGENT_NAME,
        model=settings.model,
        system_prompt=settings.system_prompt,
        skills=None,
        tools=None,
        temperature=None,
        interrupt_on=runtime_settings.interrupt_on(),
        thinking=None,
        owner=_SYSTEM_OWNER,
        builtin=True,
        description="Built-in agent (DEEPAGENTS_MODEL + SYSTEM_PROMPT env settings)",
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _value(cfg: AgentConfigIn, *, owner: str, created_at: str | None = None) -> dict[str, Any]:
    updated = _now_iso()
    return {
        "name": cfg.name,
        "description": cfg.description,
        "model": cfg.model,
        "system_prompt": cfg.system_prompt,
        "skills": cfg.skills,
        "tools": cfg.tools,
        "temperature": cfg.temperature,
        "interrupt_on": cfg.interrupt_on,
        "thinking": cfg.thinking,
        "scope": cfg.scope,
        "owner": owner,
        "created_at": created_at or updated,
        "updated_at": updated,
    }


def _to_out(value: dict[str, Any], *, builtin: bool = False) -> AgentConfigOut:
    return AgentConfigOut(**value, builtin=builtin)


def _spec_from_value(value: dict[str, Any]) -> AgentSpec:
    return AgentSpec(
        name=value["name"],
        model=value["model"],
        system_prompt=value.get("system_prompt"),
        skills=value.get("skills"),
        tools=value.get("tools"),
        temperature=value.get("temperature"),
        interrupt_on=value.get("interrupt_on"),
        thinking=value.get("thinking"),
        owner=value.get("owner") or _GLOBAL_OWNER,
        description=value.get("description"),
        created_at=value.get("created_at"),
        updated_at=value.get("updated_at"),
    )


def _skill_file_key(name: str) -> str:
    return f"/{name}/SKILL.md"


def _validate_tools(tools: list[str] | None, known_servers: list[str]) -> None:
    """Reject unknown tool server names (the 'web_search' pseudo-tool is built in)."""
    if not tools:
        return
    unknown = [t for t in tools if t != "web_search" and t not in known_servers]
    if unknown:
        raise ValueError(f"Unknown tool server(s): {', '.join(unknown)}")


# ---------------------------------------------------------------------------
# skill snapshot sync
# ---------------------------------------------------------------------------


async def _skill_source(
    store: BaseStore, skill: str, username: str, *, allow_user_skills: bool = True
) -> tuple[tuple[str, ...], str]:
    """Where a referenced skill lives: the user's own skills only.

    Skills are fully per-user — there is no global pool to fall back to.
    Returns (namespace, key) or raises ValueError when the skill is unknown.
    `allow_user_skills=False` (global agent configs) always raises: a shared
    agent cannot reference a per-user skill.
    """
    if not allow_user_skills:
        raise ValueError(
            "Global agent configs cannot reference skills — skills are per-user "
            "(create a user-scoped agent to attach skills)"
        )
    user_ns = user_skills_ns(username)
    if await store.aget(user_ns, _skill_file_key(skill)) is not None:
        return user_ns, _skill_file_key(skill)
    raise ValueError(f"Unknown skill '{skill}'")


async def sync_agent_skills(
    store: BaseStore,
    owner: str,
    name: str,
    skills: list[str] | None,
    *,
    username: str,
    allow_user_skills: bool,
) -> None:
    """Copy the referenced skills into the agent's namespace (snapshot).

    Skills resolve against the user's own skills first, then the global
    skills store. SKILL.md + every bundled file are copied, so the agent's
    SkillsMiddleware (which reads all skills under its source path) sees
    exactly the selected set. Existing copies are replaced on update;
    unlisted skills are removed.
    """
    ns = agent_skills_ns(owner, name)
    for item in await store.asearch(ns):
        await store.adelete(ns, item.key)
    if not skills:
        return
    for skill in skills:
        source_ns, md_key = await _skill_source(
            store, skill, username, allow_user_skills=allow_user_skills
        )
        item = await store.aget(source_ns, md_key)
        await store.aput(ns, md_key, item.value)
        prefix = f"/{skill}/"
        for aux in await store.asearch(source_ns):
            key = aux.key or ""
            if key.startswith(prefix) and key != md_key:
                await store.aput(ns, key, aux.value)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def get_config(store: BaseStore, name: str, username: str) -> AgentConfigOut | None:
    """Resolve a config: the user's own agent first, then global, then builtin default."""
    if name == DEFAULT_AGENT_NAME:
        return _to_out(
            {
                "name": DEFAULT_AGENT_NAME,
                "model": settings.model,
                "system_prompt": settings.system_prompt,
                "skills": None,
                "tools": None,
                "temperature": None,
                "interrupt_on": runtime_settings.interrupt_on(),
                "scope": "global",
                "owner": _SYSTEM_OWNER,
            },
            builtin=True,
        )
    item = await store.aget(user_agents_ns(username), name)
    if item is None:
        item = await store.aget(GLOBAL_AGENTS_NS, name)
    return _to_out(item.value) if item is not None else None


def _render_prompt(prompt: str | None, username: str) -> str | None:
    """Substitute per-user placeholders in a system prompt.

    Supported: ``{{username}}`` (and the aliases ``{{user}}`` /
    ``{{user_id}}``). Rendered at spec-load time so each user's graph gets
    its own prompt (the registry fingerprint covers it).
    """
    if not prompt:
        return prompt
    return (
        prompt.replace("{{username}}", username)
        .replace("{{user}}", username)
        .replace("{{user_id}}", username)
    )


async def load_spec(store: BaseStore, name: str, username: str) -> AgentSpec | None:
    """Resolve a build-ready spec (user agent -> global agent -> builtin default)."""
    if name == DEFAULT_AGENT_NAME:
        spec = default_spec()
        spec.system_prompt = _render_prompt(spec.system_prompt, username)
        return spec
    item = await store.aget(user_agents_ns(username), name)
    if item is None:
        item = await store.aget(GLOBAL_AGENTS_NS, name)
    if item is None:
        return None
    spec = _spec_from_value(item.value)
    spec.system_prompt = _render_prompt(spec.system_prompt, username)
    return spec


async def list_configs(
    store: BaseStore, username: str, *, include_global: bool = True
) -> list[AgentConfigOut]:
    """The builtin default, then the user's agents, then global agents (by name)."""
    default = await get_config(store, DEFAULT_AGENT_NAME, username)
    out: list[AgentConfigOut] = [default] if default is not None else []
    items = await store.asearch(user_agents_ns(username))
    out.extend(_to_out(it.value) for it in items if it.value)
    if include_global:
        items = await store.asearch(GLOBAL_AGENTS_NS)
        out.extend(_to_out(it.value) for it in items if it.value)
    return sorted(out, key=lambda c: (c.builtin is not True, c.name))


async def create_config(
    store: BaseStore,
    cfg: AgentConfigIn,
    username: str,
    *,
    known_servers: list[str],
) -> AgentConfigOut:
    """Create a user or global agent config (validates name, skills, tools)."""
    if cfg.name == DEFAULT_AGENT_NAME:
        raise ValueError(f"'{DEFAULT_AGENT_NAME}' is reserved")
    owner = _GLOBAL_OWNER if cfg.scope == "global" else username
    ns = GLOBAL_AGENTS_NS if cfg.scope == "global" else user_agents_ns(username)
    if (
        await store.aget(user_agents_ns(username), cfg.name) is not None
        or await store.aget(GLOBAL_AGENTS_NS, cfg.name) is not None
    ):
        raise KeyError(cfg.name)
    # User agents may reference the owner's own skills + global skills;
    # global agents only reference global skills.
    allow_user_skills = cfg.scope != "global"
    for skill in cfg.skills or []:
        await _skill_source(store, skill, username, allow_user_skills=allow_user_skills)
    _validate_tools(cfg.tools, known_servers)
    await sync_agent_skills(
        store,
        owner,
        cfg.name,
        cfg.skills,
        username=username,
        allow_user_skills=allow_user_skills,
    )
    value = _value(cfg, owner=owner)
    await store.aput(ns, cfg.name, value)
    logger.info("agent config created: name=%s scope=%s owner=%s", cfg.name, cfg.scope, owner)
    return _to_out(value)


async def update_config(
    store: BaseStore,
    name: str,
    cfg: AgentConfigIn,
    username: str,
    *,
    known_servers: list[str],
) -> AgentConfigOut:
    """Replace a config the user owns (or a global one, for admins)."""
    if cfg.name != name:
        raise ValueError("name in body must match the path")
    ns = GLOBAL_AGENTS_NS if cfg.scope == "global" else user_agents_ns(username)
    existing = await store.aget(ns, name)
    if existing is None:
        raise KeyError(name)
    allow_user_skills = cfg.scope != "global"
    for skill in cfg.skills or []:
        await _skill_source(store, skill, username, allow_user_skills=allow_user_skills)
    _validate_tools(cfg.tools, known_servers)
    owner = existing.value.get("owner") or _GLOBAL_OWNER
    await sync_agent_skills(
        store,
        owner,
        name,
        cfg.skills,
        username=username,
        allow_user_skills=allow_user_skills,
    )
    value = _value(cfg, owner=owner, created_at=existing.value.get("created_at"))
    await store.aput(ns, name, value)
    return _to_out(value)


async def delete_config(store: BaseStore, name: str, username: str, *, allow_global: bool) -> bool:
    """Delete the user's agent (or a global one when allowed); clears its skill namespace."""
    ns = GLOBAL_AGENTS_NS if allow_global else user_agents_ns(username)
    item = await store.aget(ns, name)
    if item is None:
        return False
    owner = item.value.get("owner") or _GLOBAL_OWNER
    skill_ns = agent_skills_ns(owner, name)
    for skill_item in await store.asearch(skill_ns):
        await store.adelete(skill_ns, skill_item.key)
    await store.adelete(ns, name)
    return True


__all__ = [
    "DEFAULT_AGENT_NAME",
    "AgentSpec",
    "create_config",
    "default_spec",
    "delete_config",
    "get_config",
    "list_configs",
    "load_spec",
    "sync_agent_skills",
    "update_config",
]
