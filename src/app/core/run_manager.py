"""Registry of in-flight agent runs, decoupled from their HTTP streams.

Each run is an `ActiveRun`: an asyncio task that streams the agent and
publishes events to per-subscriber queues (one per connected SSE client).
Subscribers may come and go — the run keeps executing regardless, so a client
disconnect ("new chat") never aborts the run; only `POST /threads/{id}/cancel`
does, by setting the run's stop event (the pump loop watches it and ends the
run cleanly).

Terminal events (`done`, `interrupt`) are delivered even to slow consumers by
evicting the oldest queued delta; plain deltas are dropped when a consumer's
queue is full (lossy streaming is fine — history is the durable record).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field

from .exceptions import Conflict


@dataclass
class ActiveRun:
    """One in-flight agent run and its subscriber queues."""

    thread_id: str
    username: str
    agent_name: str
    done: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)


class RunManager:
    """Tracks active runs (thread_id -> ActiveRun) and fans events out to them."""

    TERMINAL_EVENTS = frozenset({"done", "interrupt"})

    def __init__(self) -> None:
        self._runs: dict[str, ActiveRun] = {}

    def start(self, thread_id: str, username: str, agent_name: str) -> ActiveRun:
        """Register a new run; Conflict when the thread already has an active one."""
        if thread_id in self._runs:
            raise Conflict(detail=f"Thread '{thread_id}' already has an active run")
        active = ActiveRun(thread_id=thread_id, username=username, agent_name=agent_name)
        self._runs[thread_id] = active
        return active

    def get(self, thread_id: str) -> ActiveRun | None:
        """The active run of a thread (None when none or already finished)."""
        return self._runs.get(thread_id)

    def cancel(self, thread_id: str) -> bool:
        """Signal a running thread to stop; False when nothing is running.

        Idempotent guard: a second cancel on the same run returns False, so
        callers surface 409 "no active run" instead of cancelling twice.
        """
        active = self._runs.get(thread_id)
        if active is None or active.stop.is_set():
            return False
        active.stop.set()
        return True

    def is_running(self, thread_id: str) -> bool:
        """True while a run is registered on the thread (started, not finished)."""
        return thread_id in self._runs

    def subscribe(self, active: ActiveRun) -> asyncio.Queue:
        """Create a subscriber queue for one SSE client of `active`."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        active.subscribers.add(queue)
        return queue

    def unsubscribe(self, active: ActiveRun, queue: asyncio.Queue) -> None:
        """Detach a subscriber; the run keeps going regardless."""
        active.subscribers.discard(queue)

    def publish(self, active: ActiveRun, event: str, data: dict) -> None:
        """Fan an event out to all subscribers (best-effort, never blocks).

        Deltas are dropped for slow/unattached consumers; terminal events
        (done/interrupt) force their way in by evicting the oldest delta.
        """
        item = {"event": event, "data": data}
        terminal = event in self.TERMINAL_EVENTS
        for queue in list(active.subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                if not terminal:
                    continue
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    queue.put_nowait(item)

    def finish(self, thread_id: str) -> None:
        """Mark the run done and drop it from the registry.

        Call only after the terminal events were published: subscribers
        observe `done` before the flag flips, so they drain everything.
        """
        active = self._runs.pop(thread_id, None)
        if active is not None:
            active.done.set()


# Singleton shared by chat streaming and the attach/notifications endpoints.
run_manager = RunManager()
