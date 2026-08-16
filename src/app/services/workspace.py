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
  default agents (full isolation). It is the agent's authoring surface:
  write/edit skills here and the changes are synced back into the store at
  run end (``sync_skills_to_store``), then re-materialized next run.
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
import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from langgraph.store.base import BaseStore

from ..core.config import settings
from ..core.constants import user_skills_ns
from ..schema.agent_schema import SkillFileIn, SkillIn
from ..services import resources
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

# Per-user skills mirror state for the run-end writeback (guarded by the
# same per-user lock): generation counter + {relpath: sha256} of the files
# materialized by the last store -> disk sync.
_skill_generations: dict[str, int] = {}
_skill_snapshots: dict[str, tuple[int, dict[str, str]]] = {}


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
    dest.mkdir(parents=True, exist_ok=True)
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


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _walk_skill_files(folder: Path) -> dict[str, bytes]:
    """{relpath: bytes} of the regular files under ``folder`` (no symlinks)."""
    out: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for fn in filenames:
            path = Path(dirpath) / fn
            if path.is_symlink():
                continue
            out[path.relative_to(folder).as_posix()] = path.read_bytes()
    return out


def _skills_on_disk(skills_root: Path) -> dict[str, dict[str, bytes]]:
    """{name: {relpath: bytes}} for every valid skill dir under ``skills_root``.

    A skill is a directory named per SKILL_NAME_RE containing SKILL.md; the
    named-agent snapshot subtrees (``<owner>/<name>/``) never match.
    """
    out: dict[str, dict[str, bytes]] = {}
    for child in sorted(skills_root.iterdir()):
        if child.is_dir() and resources.SKILL_NAME_RE.fullmatch(child.name):
            files = _walk_skill_files(child)
            if "SKILL.md" in files:
                out[child.name] = files
    return out


def _snapshot_skill_files(snapshot: dict[str, str], name: str) -> dict[str, str]:
    """Snapshot entries under ``<name>/`` as {relpath: sha256}."""
    prefix = name + "/"
    return {rel[len(prefix) :]: h for rel, h in snapshot.items() if rel.startswith(prefix)}


async def _sync_one_skill(
    store: BaseStore,
    username: str,
    name: str,
    known: dict[str, str],
    current: dict[str, dict[str, bytes]],
) -> None:
    """Persist one skill dir into the store (create/update/delete as needed).

    ``name`` is the dir name; the store key follows the SKILL.md frontmatter
    ``name`` (same rule as publish_skill). Unchanged skills are untouched;
    bundled files the agent removed are deleted from the store.
    """
    ns = user_skills_ns(username)
    old = _snapshot_skill_files(known, name)
    files = current.get(name)
    if files is None:
        # The folder vanished since materialization: drop it from the store.
        if await resources.get_skill(store, name, ns) is not None:
            await resources.delete_skill(store, name, ns)
        return
    hashes = {rel: _file_sha256(data) for rel, data in files.items()}
    if old == hashes:
        return  # untouched since this run's materialization
    md_text = files["SKILL.md"].decode("utf-8", errors="replace")
    fname, description, body = resources.parse_skill_frontmatter(md_text)
    skill = SkillIn(
        name=fname,
        description=description,
        content=body,
        files=[
            SkillFileIn(path=rel, content=data.decode("utf-8", errors="replace"))
            for rel, data in sorted(files.items())
            if rel != "SKILL.md"
        ],
    )
    if await resources.get_skill(store, fname, ns) is not None:
        await resources.update_skill(store, fname, skill, ns, raw_markdown=md_text)
    else:
        await resources.create_skill(store, skill, ns, raw_markdown=md_text)
    # Bundled files removed by the agent since materialization.
    for rel in old:
        if rel != "SKILL.md" and rel not in hashes:
            await resources.delete_skill_file(store, fname, rel, ns)


async def sync_skills_to_store(store: BaseStore, username: str, generation: int | None) -> None:
    """Persist agent edits under the user's ``skills/`` dir into the store.

    Called at run end, the counterpart of ``materialize_skills``: the
    agent's filesystem is the authoring surface (write/edit a skill in
    ``skills/`` and the change sticks — stored server-side, visible in the
    frontend skills list, re-materialized next run), while the store stays
    the durable source of truth. Only skills that changed since this run's
    materialization are touched; unchanged skills are never rewritten.

    Scope: the user's own skills only (``skills/<name>/`` with SKILL.md).
    Named-agent snapshot subtrees (``skills/<owner>/<name>/``) are never
    written back. ``generation`` is the value returned by this run's
    ``materialize_skills``; when it is stale (another run re-materialized
    since — that run persists its own diff) the writeback is skipped.
    Best-effort like ``git_commit``: per-skill failures are logged and never
    fail the run.
    """
    if generation is None:
        return
    user = workspace_dir(username)
    skills_root = user / "skills"
    async with await _workspace_lock(username):
        snapshot = _skill_snapshots.get(username)
        if snapshot is None or snapshot[0] != generation:
            logger.info(
                "skills writeback skipped for %s (stale generation %s)", username, generation
            )
            return
        known = snapshot[1]
        current = _skills_on_disk(skills_root)
        known_names = {
            rel[: rel.index("/")]
            for rel in known
            if rel.endswith("/SKILL.md") and rel.count("/") == 1
        }
        for name in sorted(known_names | set(current)):
            try:
                await _sync_one_skill(store, username, name, known, current)
            except Exception:
                logger.exception("skills writeback failed for %s skill %r", username, name)
        # The mirror now equals the store; record it so a second writeback
        # for the same generation is a no-op.
        _skill_snapshots[username] = (
            generation,
            {rel: _file_sha256(data) for rel, data in _walk_skill_files(skills_root).items()},
        )


async def materialize_skills(
    store: BaseStore, username: str, spec: AgentSpec | None = None
) -> int | None:
    """Copy the agent's skills into the workspace.

    Called before each run. The builtin default agent gets the **user's own
    skills only** (full isolation — the admin global pool is never served to
    everyone); named agents get their snapshot namespace ([] = none). The
    target skills dir is cleared first so stale copies (e.g. from before a
    config change) can't linger. Skills are always overwritten — they are
    user/admin-owned; agent edits are synced back by ``sync_skills_to_store``
    at run end.

    Returns the snapshot generation for this materialization (None when the
    user has no skills source — nothing materialized). Pass the value to
    ``sync_skills_to_store`` at run end so the writeback only applies to
    skills as they were materialized for THIS run.
    """
    user = workspace_dir(username)
    if spec is None or spec.skills is None or spec.builtin:
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
        return None
    files = [
        (dest / (item.key or "").lstrip("/"), _value_to_bytes(item.value))
        for item in await store.asearch(ns)
    ]
    skills_root = user / "skills"
    async with await _workspace_lock(username):
        await asyncio.to_thread(_replace_skill_files, dest, files)
        # Snapshot what the mirror now contains (paths relative to skills/),
        # keyed by a generation counter: the run-end writeback diffs against
        # this snapshot and skips when another run re-materialized since.
        generation = _skill_generations.get(username, 0) + 1
        _skill_generations[username] = generation
        _skill_snapshots[username] = (
            generation,
            {path.relative_to(skills_root).as_posix(): _file_sha256(data) for path, data in files},
        )
        return generation
