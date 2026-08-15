"""User file uploads for chat: saved into the user's workspace, read by the agent.

Files posted to /chat and /api/chat (multipart) are written directly to the
user's workspace dir (``WORKSPACE_ROOT/<user_id>/uploads/``) — real files,
versioned by the workspace's git repo, no Postgres involvement. The agent is
told the virtual path (``/uploads/<name>``), which its file tools resolve
into the real workspace file.

File names are sanitized (basename only), files are size-capped, and
oversized uploads are skipped with a note instead of failing the whole chat.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..core.config import settings
from ..services.workspace import workspace_dir

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(name: str | None) -> str:
    """Basename only: strip directories, control chars, and leading dots."""
    clean = _UNSAFE.sub("_", Path(name or "").name).strip(" .")
    return clean[:120] or "upload"


async def save_upload(username: str, upload: UploadFile) -> dict[str, Any]:
    """Buffer one upload (size-capped) and write it into the user's workspace.

    Returns metadata (``name``, ``path``, ``size``, ``content_type``) or
    ``{"error": ...}`` when the file was skipped (oversized). Colliding names
    get a numeric suffix so re-uploads never clobber earlier files.
    """
    name = sanitize_filename(upload.filename)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    dest_dir = workspace_dir(username) / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        stem, suffix = name.rsplit(".", 1) if "." in name else (name, "")
        for i in range(1, 1000):
            candidate = dest_dir / f"{stem} ({i}){('.' + suffix) if suffix else ''}"
            if not candidate.exists():
                dest = candidate
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
    dest.write_bytes(content)
    return {
        "name": dest.name,
        "path": f"/uploads/{dest.name}",
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
