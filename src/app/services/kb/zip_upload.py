"""Safe zip extraction for folder uploads (one zip = many documents).

Structural safety is enforced before anything is stored:

- not a valid zip -> error
- entry count > `KB_ZIP_MAX_ENTRIES` -> error
- any entry with an unsafe path (`..`, absolute, control chars) -> error
- any entry larger than `KB_MAX_FILE_SIZE_MB` -> error
- total uncompressed size > `KB_ZIP_MAX_TOTAL_MB` -> error (zip-bomb guard)

Soft per-entry checks (extension allowlist, storage quota) stay in the
endpoint so they produce per-entry results instead of aborting the archive.
"""

from __future__ import annotations

import io
import mimetypes
from zipfile import BadZipFile, ZipFile

from ...core.config import settings
from .paths import safe_path


class ZipValidationError(ValueError):
    """The archive itself is invalid or unsafe; nothing was extracted."""


def extract_zip_entries(data: bytes) -> list[dict]:
    """Extract safe file entries from zip bytes: [{"path", "data", "mime_type"}].

    Raises:
        ZipValidationError: invalid or unsafe archive (nothing extracted).
    """
    try:
        archive = ZipFile(io.BytesIO(data))
    except BadZipFile as exc:
        raise ZipValidationError(f"Not a valid zip archive: {exc}") from exc

    infos = archive.infolist()
    if len(infos) > settings.kb_zip_max_entries:
        raise ZipValidationError(
            f"Archive has too many entries ({len(infos)} > {settings.kb_zip_max_entries})"
        )

    max_bytes = settings.kb_max_file_size_mb * 1024 * 1024
    max_total = settings.kb_zip_max_total_mb * 1024 * 1024
    total = 0
    entries: list[dict] = []
    for info in infos:
        if info.is_dir():
            continue
        path = safe_path(info.filename)
        if path is None:
            raise ZipValidationError(f"Unsafe path inside archive: {info.filename!r}")
        if info.file_size > max_bytes:
            raise ZipValidationError(
                f"Entry too large inside archive: {path} "
                f"({info.file_size} > {settings.kb_max_file_size_mb} MB)"
            )
        total += info.file_size
        if total > max_total:
            raise ZipValidationError(
                f"Archive exceeds the total size limit ({settings.kb_zip_max_total_mb} MB)"
            )
        mime_type = mimetypes.guess_type(path)[0]
        entries.append({"path": path, "data": archive.read(info), "mime_type": mime_type})
    archive.close()
    return entries
