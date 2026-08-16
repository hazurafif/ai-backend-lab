"""Per-user workspaces: the agent's only filesystem, plain host files under
``WORKSPACE_ROOT/<user_id>/``, versioned with git.

The workflow is deliberately simple: **a system prompt + the host file
system**. No virtual mounts, no store mirroring — the files on disk (a
bind-mounted host dir in compose, so ``.workspace/<user>`` shows up on the
host) are the source of truth, and the workspace root's own git repo is
versioning only.

Layout under ``WORKSPACE_ROOT/<user_id>/`` (scaffolded on user create):

- ``memories/`` — durable memory the agent reads/writes across threads
- ``skills/`` — the **user's own skills** ("my skills", /skills API),
  materialized before each run; the admin global pool is never served to
  default agents (full isolation). Agent edits are overwritten next run
- ``uploads/`` — chat uploads (written by the API, see services/uploads.py)
- ``tmp/`` — scratch space for the agent's scripts and intermediate files

The workspace root is **its own git repository**: every run end auto-commits
(all changes, best-effort). Whether to **push** is the agent's decision —
git credentials are pre-configured here (``GIT_TOKEN``/``GIT_REMOTE_URL``,
see ``ensure_git``), never used programmatically. The token lives in
``.git-credentials`` inside the workspace repo, which is gitignored so it
can never be committed. ``ensure_git`` runs at startup (lifespan), so the
repo + credentials are ready in the container before the first run.

Lifecycle: ``ensure_user_workspace`` (on user create) scaffolds the working
dirs and adds tracked ``.gitkeep`` markers; ``remove_user_workspace`` (on
user delete) removes the dir and commits. Git operations are best-effort
(never fail a run when git is unavailable).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from langgraph.store.base import BaseStore

from ..core.config import settings
from ..services.agent_configs import AgentSpec

logger = logging.getLogger(__name__)

_CREDENTIALS_FILE = ".git-credentials"
_WORKSPACE_GITIGNORE = f"{_CREDENTIALS_FILE}\n.venv/\n__pycache__/\n"
_COMMIT_PREFIX = "[AGENT] "

# Serialize store -> disk skills materialization per user: two concurrent
# runs of the same user race on the skills dir otherwise (TOCTOU between
# _replace_skill_files' rmtree/unlink and iterdir -> FileNotFoundError,
# logged as "skills materialization failed; running without it").
_workspace_locks: dict[str, asyncio.Lock] = {}
_workspace_locks_guard = asyncio.Lock()


async def _workspace_lock(username: str) -> asyncio.Lock:
    """The per-user lock guarding workspace mutations (materialization)."""
    async with _workspace_locks_guard:
        return _workspace_locks.setdefault(username, asyncio.Lock())


def workspace_root() -> Path:
    """The workspace root dir (created on demand)."""
    root = Path(settings.workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def workspace_dir(username: str) -> Path:
    """The user's workspace dir (created on demand)."""
    d = workspace_root() / username
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# git (best-effort, runs in a worker thread)
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess | None:
    """Run git in the workspace root; None when git is unavailable."""
    try:
        return subprocess.run(
            ("git", "-C", str(workspace_root()), *args),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("git unavailable (%s); workspace versioning skipped", exc)
        return None


def _configure_git() -> None:
    """Identity + credentials from settings so the agent CAN push.

    The token is stored in ``.git-credentials`` inside the workspace repo
    (gitignored, 0600) and a credential helper points git at it; the remote
    is registered when GIT_REMOTE_URL is set. Nothing here pushes.
    """
    root = workspace_root()
    _git("config", "user.name", settings.git_username)
    _git("config", "user.email", settings.git_email)
    if settings.git_remote_url:
        _git("remote", "remove", "origin")
        _git("remote", "add", "origin", settings.git_remote_url)
    if settings.git_token and settings.git_remote_url:
        host = urlparse(settings.git_remote_url).netloc
        creds = root / _CREDENTIALS_FILE
        with contextlib.suppress(OSError):
            # Un-stick a previously 0600-chmod'd file: some container bind
            # mounts (docker on macOS via OrbStack) enforce writes against the file's
            # container-visible mode regardless of uid, so a leftover 0600
            # would lock the file against this very write. Best-effort.
            creds.chmod(0o666)
        creds.write_text(f"https://{settings.git_username}:{settings.git_token}@{host}/\n")
        try:
            creds.chmod(0o600)
        except OSError:
            # Best-effort: on mounts where chmod doesn't stick (docker on
            # macOS via OrbStack) the host-side mode governs access; keep going.
            logger.warning(
                "could not chmod %s to 0600 (bind-mount limits); keeping host mode", creds
            )
        _git("config", "credential.helper", f"store --file={creds}")
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_WORKSPACE_GITIGNORE)
    else:
        # Fold in any new patterns (e.g. .venv/) for workspaces initialized
        # before they existed, so a per-user virtualenv never gets committed.
        existing = gitignore.read_text().splitlines()
        missing = [p for p in _WORKSPACE_GITIGNORE.splitlines() if p and p not in existing]
        if missing:
            gitignore.write_text("\n".join(existing + missing) + "\n")


def ensure_git() -> None:
    """Initialize the workspace repo on first use (idempotent)."""
    if not (workspace_root() / ".git").exists() and _git("init", "-q") is None:
        return
    _configure_git()
    if _git("rev-parse", "--verify", "-q", "HEAD") is None:
        _git("commit", "-q", "--allow-empty", "-m", f"{_COMMIT_PREFIX}workspace initialized")


def _git_commit(message: str) -> None:
    """Stage everything and commit with the [AGENT] prefix; no-op when unchanged."""
    ensure_git()
    if _git("add", "-A") is None:
        return
    _git("commit", "-q", "-m", f"{_COMMIT_PREFIX}{message}")


async def git_commit(message: str) -> None:
    """Commit the whole workspace (best-effort, threaded)."""
    try:
        await asyncio.to_thread(_git_commit, message)
    except Exception:
        logger.exception("workspace git commit failed")


# The per-user working directories the agent knows from its system prompt.
# Scaffolded on user create (each with a tracked .gitkeep) so the layout is
# stable on disk — and visible on the host — even before any run. Keep in
# sync with the "workspace layout" section of DEFAULT_SYSTEM_PROMPT.
WORKSPACE_SUBDIRS = ("memories", "skills", "uploads", "tmp")

# Starter memory file, auto-loaded into the system prompt before every run
# (deepagents MemoryMiddleware via MEMORY_SOURCE). HTML comments are stripped
# before injection, so the lines below are authoring notes only.
_STARTER_MEMORY = """\
<!-- This file is your long-term memory. It is loaded into your system prompt
at the start of every run. Edit it with edit_file to persist what you learn
about the user and how you should work. Keep it concise: only durable,
high-signal facts and preferences. -->

# Workspace conventions

- File tools use VIRTUAL paths (/uploads, /tmp, /memories, /skills); the
  shell sees the real filesystem with your workspace as cwd — use relative
  paths in shell commands, never /app/... with the file tools.
- Python: use `uv`; keep a .venv inside the workspace (gitignored) and
  never install into the container's global Python.
- All shell work stays inside your own user-scoped workspace dir.

# Memory

<!-- Add durable notes below, e.g.:
- user prefers concise answers with code examples
- project conventions: ...
-->
"""


def ensure_user_workspace(username: str) -> None:
    """Create the user's workspace dir + working subdirs, tracked in git.

    Scaffolds ``memories/``, ``skills/``, ``uploads/`` and ``tmp/`` (each
    with a tracked ``.gitkeep``) so the agent's working directories exist on
    the host right after user creation — no run needed to materialize them.
    """
    d = workspace_dir(username)
    root_marker = d / ".gitkeep"
    if not root_marker.exists():
        root_marker.write_text("")
    for name in WORKSPACE_SUBDIRS:
        sub = d / name
        sub.mkdir(parents=True, exist_ok=True)
        marker = sub / ".gitkeep"
        if not marker.exists():
            marker.write_text("")
    # Auto-loaded memory file (see MEMORY_SOURCE in core/constants.py).
    memory_file = d / "memories" / "AGENTS.md"
    if not memory_file.exists():
        memory_file.write_text(_STARTER_MEMORY)
    if _git("add", "-A") is not None:
        _git_commit(f"create workspace for user {username}")


def remove_user_workspace(username: str) -> None:
    """Remove the user's workspace dir and commit (user delete)."""
    d = workspace_root() / username
    if d.exists():
        shutil.rmtree(d)
    if _git("add", "-A") is not None:
        _git_commit(f"delete workspace for user {username}")


# ---------------------------------------------------------------------------
# skills materialization (the only store -> disk sync)
# ---------------------------------------------------------------------------


def _value_to_bytes(value: dict) -> bytes:
    """Store item value -> file bytes (utf-8 text or base64 binary)."""
    content = value.get("content") or ""
    if isinstance(content, list):  # legacy format
        content = "\n".join(content)
    if value.get("encoding") == "base64":
        import base64

        return base64.standard_b64decode(content)
    return str(content).encode("utf-8")


def _replace_skill_files(dest: Path, files: list[tuple[Path, bytes]]) -> None:
    """Clear the skills dir and write the fresh materialization."""
    if dest.exists():
        for child in dest.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for target, data in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Keep the dir git-tracked when it ends up empty (no skills yet).
    if not any(dest.iterdir()) and not (dest / ".gitkeep").exists():
        (dest / ".gitkeep").write_text("")


async def materialize_skills(
    store: BaseStore, username: str, spec: AgentSpec | None = None
) -> None:
    """Copy the agent's skills into the workspace.

    Called before each run. The builtin default agent gets the **user's own
    skills only** (full isolation — the admin global pool is never served to
    everyone); named agents get their snapshot namespace ([] = none). The
    target skills dir is cleared first so stale copies (e.g. from before a
    config change) can't linger. Skills are always overwritten — they are
    user/admin-owned; agent edits are discarded next run.
    """
    user = workspace_dir(username)
    if spec is None or spec.skills is None or spec.builtin:
        from ..core.constants import user_skills_ns

        ns = user_skills_ns(username)
        dest = user / "skills"
    elif spec.skills_source:
        from ..core.constants import agent_skills_ns

        # Source is /skills/<owner>/<name>/; the middleware reads the same
        # virtual path, which the workspace backend maps under skills/.
        _prefix, owner, name = spec.skills_source.strip("/").split("/")
        ns = agent_skills_ns(owner, name)
        dest = user / "skills" / owner / name
    else:
        return
    files = [
        (dest / (item.key or "").lstrip("/"), _value_to_bytes(item.value))
        for item in await store.asearch(ns)
    ]
    async with await _workspace_lock(username):
        await asyncio.to_thread(_replace_skill_files, dest, files)
