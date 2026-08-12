"""Registry of active agent runs, keyed by thread_id.

`agent_stream` registers a stop event per thread before pumping the stream;
`POST /threads/{id}/cancel` sets that event so the run aborts cleanly and
the client gets a terminal `done` event. Stale entries are removed when a
run finishes (unregister) or when a new run starts on the same thread.
"""

from __future__ import annotations

import asyncio


class RunRegistry:
    """Maps thread_id -> asyncio.Event signalling "stop this run now"."""

    def __init__(self) -> None:
        self._stops: dict[str, asyncio.Event] = {}

    def register(self, thread_id: str) -> asyncio.Event:
        """Create (or reset) the stop event for a thread and return it."""
        event = asyncio.Event()
        self._stops[thread_id] = event
        return event

    def unregister(self, thread_id: str) -> None:
        self._stops.pop(thread_id, None)

    def cancel(self, thread_id: str) -> bool:
        """Signal a running thread to stop; False when nothing is running."""
        event = self._stops.get(thread_id)
        if event is None or event.is_set():
            return False
        event.set()
        return True

    def is_running(self, thread_id: str) -> bool:
        return thread_id in self._stops


# Singleton shared by agent_stream and the cancel endpoint.
runs = RunRegistry()
