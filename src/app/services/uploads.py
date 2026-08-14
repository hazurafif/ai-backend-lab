"""User file uploads for chat: saved to disk, manipulated by the agent's tools.

Files posted to /chat and /api/chat (multipart) land under
``<UPLOADS_DIR>/<username>/`` — the agent's filesystem root is the server cwd
when ``EXECUTE_ENABLED=true``, so the agent can read and manipulate them with
its own tools (``read_file``, ``execute`` with e.g. ``pdftotext`` for PDFs)
instead of the API trying to parse arbitrary formats. The endpoint appends the
exact paths to the user message so the agent knows what it can work with.

File names are sanitized (basename only), files are size-capped, and
oversized uploads are skipped with a note instead of failing the whole chat.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..core.config import settings

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(name: str | None) -> str:
    """Basename only: strip directories, control chars, and leading dots."""
    clean = _UNSAFE.sub("_", Path(name or "").name).strip(" .")
    return clean[:120] or "upload"


def agent_path(dest: Path) -> str:
    """Path as the agent's tools see it (relative to the server cwd)."""
    try:
        return dest.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(dest)


def upload_dir(username: str) -> Path:
    """Per-user upload directory (created on demand)."""
    return Path(settings.uploads_dir) / username


async def save_upload(username: str, upload: UploadFile) -> dict[str, Any]:
    """Stream one upload to disk, size-capped.

    Returns metadata (``name``, ``path``, ``size``, ``content_type``) or
    ``{"error": ...}`` when the file was skipped (oversized). Colliding names
    get a numeric suffix so re-uploads never clobber earlier files.
    """
    name = sanitize_filename(upload.filename)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    dest_dir = upload_dir(username)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        stem, suffix = name.rsplit(".", 1) if "." in name else (name, "")
        for i in range(1, 1000):
            candidate = dest_dir / f"{stem} ({i}){('.' + suffix) if suffix else ''}"
            if not candidate.exists():
                dest = candidate
                break
    size = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    dest.unlink(missing_ok=True)
                    return {
                        "error": f"{name} (skipped: exceeds {settings.max_upload_size_mb} MB cap)"
                    }
                fh.write(chunk)
    finally:
        await upload.close()
    return {
        "name": dest.name,
        "path": agent_path(dest),
        "size": size,
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
