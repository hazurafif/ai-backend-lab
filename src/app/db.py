"""Persistence layer: Postgres checkpointer + store, with in-memory fallback.

- Checkpointer (threads/conversation state) -> AsyncPostgresSaver
- Store (long-term memory, thread metadata)   -> PostgresStore
Both come from `langgraph-checkpoint-postgres`. When `DATABASE_URI` is unset
(no Postgres available), we fall back to `InMemorySaver` + `InMemoryStore` so
the backend still runs locally for development.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Checkpointer

from .core.config import settings

logger = logging.getLogger(__name__)


class Persistence:
    """Owns the checkpointer and store, set up at app startup and closed at shutdown."""

    def __init__(self) -> None:
        self.checkpointer: Checkpointer | None = None
        self.store: BaseStore | None = None
        self.backend_name = "memory"
        self._saver_cm = None
        self._store_cm = None

    @property
    def is_postgres(self) -> bool:
        return self.backend_name == "postgres"

    async def start(self) -> None:
        if settings.database_uri:
            try:
                # AsyncPostgresSaver / PostgresStore are async context managers.
                self._saver_cm = AsyncPostgresSaver.from_conn_string(settings.database_uri)
                self.checkpointer = await self._saver_cm.__aenter__()
                await self.checkpointer.setup()  # create tables

                self._store_cm = AsyncPostgresStore.from_conn_string(settings.database_uri)
                self.store = await self._store_cm.__aenter__()
                await self.store.setup()

                self.backend_name = "postgres"
                logger.info("Persistence: Postgres (checkpointer + store) connected")
                return
            except Exception:
                logger.exception(
                    "Failed to connect to Postgres (%s); falling back to in-memory",
                    settings.database_uri,
                )
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.backend_name = "memory"
        logger.warning("Persistence: in-memory (set DATABASE_URI for Postgres)")

    async def stop(self) -> None:
        for cm in (self._saver_cm, self._store_cm):
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error closing Postgres connection")
        self._saver_cm = self._store_cm = None


# Singleton, initialized in FastAPI lifespan.
persistence = Persistence()
