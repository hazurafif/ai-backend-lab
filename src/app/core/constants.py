"""App-wide constants: store namespaces, backend sources, SSE headers."""

from __future__ import annotations

# Backend source where store-backed skills live (SkillsMiddleware source path).
SKILLS_SOURCE = "/skills/"

# Store namespaces for agent-level (global, shared by all users) resources.
GLOBAL_SKILLS_NS = ("agent", "skills")
TOOL_SERVERS_NS = ("agent", "mcp_servers")


def user_skills_ns(username: str) -> tuple[str, ...]:
    """Store namespace for a user's own skills ("my skills")."""
    return ("user", "skills", username)


# Agent configs (customizable agent profiles): global agents are shared by
# all users; user agents live under ("agents", <username>).
GLOBAL_AGENTS_NS = ("agent", "agents")
DEFAULT_AGENT_NAME = "default"


def user_agents_ns(username: str) -> tuple[str, ...]:
    """Store namespace for a user's own agent configs."""
    return ("agents", username)


def agent_skills_ns(owner: str, name: str) -> tuple[str, ...]:
    """Store namespace for a named agent's skills (snapshot copies).

    `owner` is the username for user agents, "global" for global agents.
    Kept under a distinct top-level segment so prefix searches on
    GLOBAL_SKILLS_NS never see per-agent copies.
    """
    return ("agent", "agent_skills", owner, name)


def agent_skills_source(owner: str, name: str) -> str:
    """Filesystem source path a named agent's SkillsMiddleware reads."""
    return f"{SKILLS_SOURCE}{owner}/{name}/"


# Share links: key = unguessable share token, value =
# {"thread_id", "username", "created_at"} — served publicly by GET /shared/{token}.
SHARE_NS = ("shared",)


def thread_metadata_ns(username: str) -> tuple[str, ...]:
    """Store namespace for a user's thread metadata (title, timestamps, share_token)."""
    return ("threads", username)


# SSE streaming headers: disable buffering so events reach the client live.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant running inside a backend service.
You can:
- plan multi-step work and delegate subtasks to subagents (use the `task` tool)
- read/write/edit files in your workspace
- use MCP tools exposed by connected servers (they may return structured data
  that the frontend renders as interactive UI elements)
- remember things across conversations in your memory files (under memories/)
- filesystem: everything you see is real and per-user — file tools and the
  execute tool agree on the same files. Your workspace is
  .workspace/{{username}} and your shell cwd is that dir; absolute tool
  paths (/memories/, /skills/, /uploads/, /anything) resolve inside it —
  use relative paths in commands. Treat your workspace as your entire
  world: never read, list or modify other users' directories (each user
  has one dir under the workspace root — only yours belongs to you).
  Skills are in skills/ (read-only — copy one out to edit or run it);
  everything else you write is persisted on disk and auto-committed to the
  workspace git repo when the run ends
- versioning: your workspace is a git repository, auto-committed after each
  run. If a remote is configured you may `git push` yourself when you judge
  it useful — the files on disk are always the source of truth

Be concise and direct. When you call tools, explain what you are doing in one
short line so the user can follow along in the live stream.
"""
