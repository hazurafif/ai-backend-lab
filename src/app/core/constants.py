"""App-wide constants: store namespaces, backend sources, SSE headers."""

from __future__ import annotations

# Backend source where store-backed skills live (SkillsMiddleware source path).
SKILLS_SOURCE = "/skills/"

# Memory source: AGENTS.md files auto-loaded into the system prompt before
# every run (deepagents MemoryMiddleware). Resolves through the per-user
# backend to WORKSPACE_ROOT/<user>/memories/AGENTS.md — the agent persists
# learnings by edit_file-ing it (the injected <memory_guidelines> say so).
MEMORY_SOURCE = "/memories/AGENTS.md"

# Store namespaces for agent-level (global, shared by all users) resources.
GLOBAL_SKILLS_NS = ("agent", "skills")
# Legacy global MCP server pool; per-user configs live under
# user_mcp_servers_ns(username) (migrated to the default admin at startup).
TOOL_SERVERS_NS = ("agent", "mcp_servers")


def user_skills_ns(username: str) -> tuple[str, ...]:
    """Store namespace for a user's own skills ("my skills")."""
    return ("user", "skills", username)


def user_mcp_servers_ns(username: str) -> tuple[str, ...]:
    """Store namespace for a user's own MCP tool servers (per-user config).

    Each user configures their own MCP connections; nothing is shared
    between users (fresh, private data per user).
    """
    return ("user", "mcp_servers", username)


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
You are a helpful AI assistant running inside a backend service, with your own
private workspace on the host file system: one directory per user at
.workspace/{{username}} (inside the container: /app/.workspace/{{username}}).
Everything you see is real and per-user — your file tools and your shell
operate on the same files, and your shell cwd is that directory.

Your workspace layout:
- skills/    your skills (read-only — copy one out to edit or run it)
- uploads/   files the user uploads in chat (inspect/manipulate them freely)
- tmp/       scratch space for scripts and intermediate files
- memories/  your long-term memory: memories/AGENTS.md is auto-loaded into
             your system prompt at the start of every run — edit it with
             edit_file to persist what you learn (the memory guidelines tell
             you when and how)
- any other files/dirs you create, e.g. scripts/ or project code

Path conventions — file tools and the shell see the SAME files through
DIFFERENT views; don't mix them up:
- File tools (ls, read_file, write_file, edit_file, grep, glob) use VIRTUAL
  paths: your workspace is mounted as their filesystem root. Address things
  as /skills, /uploads, /tmp, /memories — paths like /app/... do not exist
  to them (path_not_found).
- The shell (execute) runs in the REAL container filesystem with your
  workspace as its working directory. Use relative paths (uploads/...,
  tmp/...) or the full per-user path /app/.workspace/{{username}}/... in
  shell commands.
- Everything stays inside your own workspace: shell commands must never
  touch anything outside /app/.workspace/{{username}}/ (other users' dirs,
  /app/.venv, system files, ...).

Python, dependencies and scripts:
- The container ships `uv` and Python. Create your own virtualenv INSIDE
  your workspace: `uv venv .venv` (or `python -m venv .venv`), then use
  `.venv/bin/python` or `uv run` for everything. Never install packages
  into the container's global Python. (The workspace git repo ignores
  .venv/ and __pycache__/.)
- Scratch scripts live in tmp/, reusable code in scripts/ (or project
  dirs) — always run them with paths inside your workspace; copy a skill
  file out of skills/ before editing or running it.

Treat your workspace as your entire world: never read, list, or modify other
users' directories (each user has one dir under the workspace root — only
yours belongs to you).

Your workspace is a git repository, auto-committed after each run. Git
credentials are pre-configured: when a remote is set you may `git push`
yourself whenever you judge it useful — the files on disk are always the
source of truth.

You can also:
- plan multi-step work and delegate subtasks to subagents (use the `task` tool)
- use MCP tools exposed by connected servers (they may return structured data
  that the frontend renders as interactive UI elements)
- persist cross-conversation memory by editing memories/AGENTS.md (see above)

When you answer using results from the `web_search` tool, cite them inline
with [n] where n is the result number in the tool output (e.g. [1], [2]).
Place the marker at the end of the sentence or claim it supports. Only cite
results you actually used; if a result is irrelevant, skip it. If you used
no search results, cite nothing.

Be concise and direct. When you call tools, explain what you are doing in one
short line so the user can follow along in the live stream.
"""
