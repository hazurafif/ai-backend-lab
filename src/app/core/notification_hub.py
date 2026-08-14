"""Per-user run lifecycle notifications (in-process pub/sub).

Runs publish lifecycle events (`run_started`, `run_completed`, ...) to the
owning user's channel. Each user has a small ring buffer of recent events
(replay on connect / REST listing) and a set of subscriber queues fed by
`GET /notifications/stream` SSE connections. All operations are synchronous
and await-free: single-threaded asyncio gives publish/subscribe atomicity.

Note: in-process only — events do not cross worker processes (same limitation
as `run_registry` / `run_manager`).
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from contextlib import suppress

RECENT_LIMIT = 200  # per-user ring buffer of lifecycle events


class NotificationHub:
    """Per-user fan-out of run lifecycle events with replay support."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._recent: dict[str, deque[dict]] = {}
        self._seq: dict[str, int] = {}

    def publish(self, username: str, event: dict) -> None:
        """Append `event` to the user's ring buffer and fan it out.

        Slow subscribers drop events (their stream is lossy by design; the
        REST endpoint + replay buffer are the durable view).
        """
        seq = self._seq.get(username, 0) + 1
        self._seq[username] = seq
        event = {"seq": seq, "event_id": uuid.uuid4().hex, **event}
        self._recent.setdefault(username, deque(maxlen=RECENT_LIMIT)).append(event)
        for queue in list(self._subscribers.get(username, ())):
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def subscribe(self, username: str, since: int | None = None) -> asyncio.Queue:
        """Create a subscriber queue, seeded with the user's recent events.

        `since` (an event seq) skips already-seen events on reconnect.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        for event in self._recent.get(username, ()):
            if since is not None and event.get("seq", 0) <= since:
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                break
        self._subscribers.setdefault(username, set()).add(queue)
        return queue

    def unsubscribe(self, username: str, queue: asyncio.Queue) -> None:
        """Detach a subscriber (connection closed)."""
        subscribers = self._subscribers.get(username)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(username, None)

    def recent(self, username: str, limit: int = 50) -> list[dict]:
        """The user's most recent lifecycle events, newest first."""
        events = list(self._recent.get(username, ()))
        events.reverse()
        return events[:limit]

    def reset(self) -> None:
        """Drop all subscribers, buffers and sequence counters (tests)."""
        self._subscribers.clear()
        self._recent.clear()
        self._seq.clear()


# Singleton shared by chat runs and the notifications endpoints.
hub = NotificationHub()
