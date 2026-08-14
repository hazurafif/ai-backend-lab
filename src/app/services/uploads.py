"""User file uploads for chat: saved to the store, read from the workspace.

Files posted to /chat and /api/chat (multipart) are written to the LangGraph
store (Postgres in production) at the key the workspace sync maps to disk:
``uploads/<name>`` under the user's workspace dir on the next run. The agent
is told the virtual path (``/uploads/<name>``), which its file tools resolve
into the real workspace file — no host-disk copy, no repo pollution.

File names are sanitized (basename only), files are size-capped, and
oversized uploads are skipped with a note instead of failing the whole chat.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from deepagents.backends.store import StoreBackend
from fastapi import UploadFile

from ..core.config import settings
from ..core.database import persistence

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(name: str | None) -> str:
    """Basename only: strip directories, control chars, and leading dots."""
    clean = _UNSAFE.sub("_", Path(name or "").name).strip(" .")
    return clean[:120] or "upload"


def _persist_to_store(username: str, name: str, data: bytes) -> None:
    """Write the upload into the store at the workspace-mapped key.

    The workspace sync copies keys ``/<username>/<name>`` in the
    ``(username,)`` namespace to ``uploads/<name>`` in the user's workspace
    dir at run start, so the agent reads ``/uploads/<name>``. No-op when the
    store is unavailable (e.g. persistence not started).
    """
    store = persistence.store
    if store is None:
        return
    backend = StoreBackend(store=store, namespace=lambda _rt, u=username: (u,))
    backend.upload_files([(f"/{username}/{name}", data)])


async def save_upload(username: str, upload: UploadFile) -> dict[str, Any]:
    """Buffer one upload (size-capped) and persist it to the store.

    Returns metadata (``name``, ``path``, ``size``, ``content_type``) or
    ``{"error": ...}`` when the file was skipped (oversized). Colliding names
    get a numeric suffix so re-uploads never clobber earlier files.
    """
    name = sanitize_filename(upload.filename)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    # Colliding names get a numeric suffix so re-uploads never clobber
    # earlier files (dedupe against the store's upload keys).
    upload_prefix = f"/{username}/"
    existing = {
        (it.key or "")[len(upload_prefix) :]
        for it in await persistence.store.asearch((username,))
        if (it.key or "").startswith(upload_prefix)
    }
    if name in existing:
        stem, suffix = name.rsplit(".", 1) if "." in name else (name, "")
        for i in range(1, 1000):
            candidate = f"{stem} ({i}){('.' + suffix) if suffix else ''}"
            if candidate not in existing:
                name = candidate
                break
    data = bytearray()
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            data += chunk
            if len(data) > max_bytes:
                return {"error": f"{name} (skipped: exceeds {settings.max_upload_size_mb} MB cap)"}
    finally:
        await upload.close()
    content = bytes(data)
    _persist_to_store(username, name, content)
    return {
        "name": name,
        "path": f"/uploads/{name}",
        "size": len(content),
        "content_type": upload.content_type or "",
    }


async def save_uploads(username: str, files: list[UploadFile]) -> list[dict[str, Any]]:
    """Save all uploads in one request; errors become per-file notes."""
    return [await save_upload(username, f) for f in files]


def file_notes(results: list[dict[str, Any]]) -> str:
    """Message text describing the uploaded files for the agent.

    Empty when nothing was uploaded. The agent uses these paths with its
    filesystem/execute tools (``pdftotext``, ``python``, ...) to inspect and
    manipulate the files.
    """
    if not results:
        return ""
    lines = [
        "Uploaded files (available in your workspace; inspect/manipulate them "
        "with your filesystem and execute tools — e.g. `pdftotext` for PDFs):"
    ]
    for r in results:
        if "error" in r:
            lines.append(f"- {r['error']}")
            continue
        size_kb = r["size"] / 1024
        size_txt = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
        ctype = f" ({r['content_type']})" if r["content_type"] else ""
        lines.append(f"- {r['path']}{ctype}, {size_txt}")
    return "\n".join(lines)
