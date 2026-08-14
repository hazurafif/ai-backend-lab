"""Workspace sync: materialize store files into the per-user workspace dir.

Model: the LangGraph store (Postgres in production) is the durable source of
truth for memories, uploads and skills; before each agent run the user's
workspace dir (``WORKSPACE_ROOT/<user_id>/``) is materialized from it, and
after the run the agent's changes are synced back. The agent only ever sees
real files — file tools and the ``execute`` tool agree — so skill scripts are
directly executable (no virtual mounts).

Layout under ``WORKSPACE_ROOT/<user_id>/``:

- ``memories/`` — store ns ``(user,)``, keys ``/<name>``. Agent-owned:
  synced down only when the disk file is missing (after a successful run the
  disk copy is at least as fresh as the store), always synced up.
- ``uploads/`` — store ns ``(user,)``, keys ``/<user>/<name>``. API-owned:
  always synced down (idempotent), always synced up.
- ``skills/`` — global skills ns (and named-agent snapshots). Admin-owned:
  always synced down, **never** synced up (agent edits to skills are
  discarded).

Deletions are never propagated from disk to the store (a file the agent
deletes comes back from the store on the next run — safer than losing it).
Disk IO runs in a worker thread; store calls are async.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

from langgraph.store.base import BaseStore

from ..core.config import settings
from ..services.agent_configs import AgentSpec

logger = logging.getLogger(__name__)


def workspace_dir(username: str) -> Path:
    """The user's real workspace dir (created on demand)."""
    root = Path(settings.workspace_root).resolve()
    d = root / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def _value_to_bytes(value: dict) -> bytes:
    """Store item value -> file bytes (utf-8 text or base64 binary)."""
    content = value.get("content") or ""
    if isinstance(content, list):  # legacy format
        content = "\n".join(content)
    if value.get("encoding") == "base64":
        return base64.standard_b64decode(content)
    return str(content).encode("utf-8")


def _bytes_to_value(data: bytes) -> dict:
    """File bytes -> store item value (utf-8 text or base64 binary)."""
    try:
        return {"content": data.decode("utf-8"), "encoding": "utf-8"}
    except UnicodeDecodeError:
        return {
            "content": base64.standard_b64encode(data).decode("ascii"),
            "encoding": "base64",
        }


def _write_items(targets: list[tuple[Path, bytes]], *, skip_existing: bool) -> None:
    """Write store items to disk (blocking; run in a worker thread)."""
    for target, data in targets:
        if skip_existing and target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


async def _copy_ns_to_dir(
    store: BaseStore, ns: tuple[str, ...], dest_dir: Path, *, skip_existing: bool
) -> None:
    """Copy every item in a store namespace into a directory (key -> rel path)."""
    items = [
        (dest_dir / (it.key or "").lstrip("/"), _value_to_bytes(it.value))
        for it in await store.asearch(ns)
    ]
    await asyncio.to_thread(_write_items, items, skip_existing=skip_existing)


def _scan_files(src_dir: Path) -> list[tuple[Path, bytes]]:
    """Every file under a directory as (path, bytes) (blocking; thread)."""
    if not src_dir.is_dir():
        return []
    return [(f, f.read_bytes()) for f in sorted(src_dir.rglob("*")) if f.is_file()]


async def _sync_dir_to_ns(
    store: BaseStore,
    src_dir: Path,
    ns: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
    key_prefix: str = "",
) -> None:
    """Upsert every file in a directory into a store namespace (rel path -> key)."""
    for file, data in await asyncio.to_thread(_scan_files, src_dir):
        rel = file.relative_to(src_dir)
        if rel.parts and rel.parts[0] in exclude:
            continue
        key = f"{key_prefix}/{rel.as_posix()}"
        await store.aput(ns, key, _bytes_to_value(data))


async def sync_down(store: BaseStore, username: str, spec: AgentSpec | None = None) -> None:
    """Materialize the user's store files into their workspace dir.

    Called before each run. `spec` selects which skills to materialize: the
    global source for the builtin default agent, the agent's snapshot
    namespace for named agents with a skill selection ([] = none).
    """
    user = workspace_dir(username)
    # Memories + uploads share the (user,) namespace; upload keys start with
    # /<username>/ — memories are everything else.
    upload_prefix = f"/{username}/"
    memories: list[tuple[Path, bytes]] = []
    uploads: list[tuple[Path, bytes]] = []
    for item in await store.asearch((username,)):
        key = item.key or ""
        data = _value_to_bytes(item.value)
        if key.startswith(upload_prefix):
            uploads.append((user / "uploads" / key[len(upload_prefix) :], data))
        else:
            memories.append((user / "memories" / key.lstrip("/"), data))
    await asyncio.to_thread(_write_items, memories, skip_existing=True)
    await asyncio.to_thread(_write_items, uploads, skip_existing=False)
    # Skills: admin-owned, always refreshed, never synced back.
    if spec is None or spec.skills is None or spec.builtin:
        await _copy_ns_to_dir(store, ("agent", "skills"), user / "skills", skip_existing=False)
    elif spec.skills_source:
        from ..core.constants import agent_skills_ns

        # Source is /skills/<owner>/<name>/; the middleware reads the same
        # virtual path, which the workspace backend maps under skills/.
        _prefix, owner, name = spec.skills_source.strip("/").split("/")
        await _copy_ns_to_dir(
            store,
            agent_skills_ns(owner, name),
            user / "skills" / owner / name,
            skip_existing=False,
        )


async def sync_up(store: BaseStore, username: str) -> None:
    """Persist the agent's workspace changes back to the store.

    Called after every run (success, error, cancel, disconnect). The whole
    user dir except ``skills/`` syncs back: root files and ``memories/`` map
    to keys ``/<name>`` in the ``(user,)`` namespace, ``uploads/`` to
    ``/<user>/<name>``. Skills are admin-owned and deletions are never
    propagated.
    """
    user = workspace_dir(username)
    try:
        # Root-level files (e.g. /script.py) sync to keys /<name>; the
        # managed subdirs are handled below with their own key shapes.
        await _sync_dir_to_ns(store, user, (username,), exclude=("memories", "uploads", "skills"))
    except Exception:
        logger.exception("workspace sync-up failed for %s", username)
    try:
        await _sync_dir_to_ns(store, user / "memories", (username,))
    except Exception:
        logger.exception("memories sync-up failed for %s", username)
    try:
        await _sync_dir_to_ns(store, user / "uploads", (username,), key_prefix=f"/{username}")
    except Exception:
        logger.exception("uploads sync-up failed for %s", username)
