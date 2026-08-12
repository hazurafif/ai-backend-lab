"""Chat sharing: token-gated public read-only views of a thread.

A share token is an unguessable random string (secrets.token_urlsafe). The
token is stored both on the thread's metadata (so the owner can see and
revoke it) and under the global ``SHARE_NS`` namespace keyed by the token
itself (so ``GET /shared/{token}`` resolves without knowing the owner).
Revoking or deleting the thread removes both entries.
"""

from __future__ import annotations

import secrets
from typing import Any

from ..core.constants import SHARE_NS, thread_metadata_ns
from ..core.database import persistence
from ..core.exceptions import NotFound
from ..util.date import now_iso


async def create_share(thread_id: str, username: str) -> dict[str, Any]:
    """Create a share token for a thread; idempotent — returns the existing token when already shared."""
    ns = thread_metadata_ns(username)
    item = await persistence.store.aget(ns, thread_id)
    if item is None:
        raise NotFound(detail=f"Thread '{thread_id}' not found")

    value = dict(item.value)
    share_token = value.get("share_token")
    if not share_token:
        share_token = secrets.token_urlsafe(32)
        value["share_token"] = share_token
        await persistence.store.aput(ns, thread_id, value)

    existing = await persistence.store.aget(SHARE_NS, share_token)
    if existing is None:
        await persistence.store.aput(
            SHARE_NS,
            share_token,
            {"thread_id": thread_id, "username": username, "created_at": now_iso()},
        )
    return {"share_token": share_token}


async def get_share(thread_id: str, username: str) -> dict[str, Any] | None:
    """The thread's share token, or None when the thread is not shared."""
    item = await persistence.store.aget(thread_metadata_ns(username), thread_id)
    if item is None:
        return None
    share_token = item.value.get("share_token")
    if not share_token:
        return None
    return {"share_token": share_token}


async def revoke_share(thread_id: str, username: str) -> bool:
    """Remove the thread's share link; returns False when it had none."""
    ns = thread_metadata_ns(username)
    item = await persistence.store.aget(ns, thread_id)
    if item is None:
        raise NotFound(detail=f"Thread '{thread_id}' not found")
    share_token = item.value.get("share_token")
    if not share_token:
        return False
    value = dict(item.value)
    del value["share_token"]
    await persistence.store.aput(ns, thread_id, value)
    await persistence.store.adelete(SHARE_NS, share_token)
    return True


async def revoke_by_thread(thread_id: str, username: str, share_token: str | None) -> None:
    """Cleanup helper for thread deletion: drop the share entry if present."""
    if share_token:
        await persistence.store.adelete(SHARE_NS, share_token)


async def lookup_share(share_token: str) -> dict[str, Any] | None:
    """Resolve a share token to its {\"thread_id\", \"username\", \"created_at\"} entry."""
    item = await persistence.store.aget(SHARE_NS, share_token)
    if item is None:
        return None
    return dict(item.value)
