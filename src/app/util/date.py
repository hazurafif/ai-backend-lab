"""Small shared utilities (time helpers)."""

from __future__ import annotations

from datetime import UTC, datetime


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()
