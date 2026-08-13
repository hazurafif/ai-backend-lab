"""Path validation for KB uploads (multipart `path` fields and zip entries).

Keeps client-supplied relative paths inside the knowledge base: no absolute
paths, no `..` traversal, no control characters, sane length.
"""

from __future__ import annotations

_MAX_PATH_LEN = 500


def safe_path(raw: str | None) -> str | None:
    """Normalize a client-supplied relative path; None when unsafe/invalid."""
    if not raw:
        return None
    path = raw.replace("\\", "/").strip().lstrip("/")
    if not path or len(path) > _MAX_PATH_LEN:
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(ch in path for ch in "\x00\r\n\t"):
        return None
    return "/".join(parts)
