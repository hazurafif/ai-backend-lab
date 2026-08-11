"""Persistence layer: Postgres checkpointer + store + chat history, in-memory fallback.

- Checkpointer (threads/conversation state) -> AsyncPostgresSaver
- Store (long-term memory, thread metadata)   -> PostgresStore
- Chat history (readable message rows)        -> chat_messages table
All come from `langgraph-checkpoint-postgres` / `psycopg`. When `DATABASE_URI`
is unset (no Postgres available), we fall back to `InMemorySaver` +
`InMemoryStore` so the backend still runs locally for development.
"""

from __future__ import annotations

import json
import logging
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Checkpointer
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ..util.date import now_iso
from .config import settings

logger = logging.getLogger(__name__)


CHAT_MESSAGES_DDL = """\
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    username    TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (thread_id, message_id)
);
CREATE INDEX IF NOT EXISTS ix_chat_messages_thread
    ON chat_messages (thread_id, created_at);
"""


class ChatHistoryStore:
    """Chat history as plain rows in Postgres (the `chat_messages` table).

    One row per message: role + serialized LangChain message + timestamp, so
    history is readable/queryable via SQL instead of opaque checkpoint blobs.
    Writes are deduped by message id, so re-runs (resume, retries) never
    duplicate rows. Falls back to an in-memory dict when Postgres is off.
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._memory: dict[str, list[dict]] = {}
        self._memory_ids: dict[str, set[str]] = {}

    @property
    def is_postgres(self) -> bool:
        return self._pool is not None

    async def start(self, uri: str | None) -> None:
        """Open the pool and create the table; in-memory when Postgres is off."""
        self._memory, self._memory_ids = {}, {}
        if uri:
            try:
                pool = AsyncConnectionPool(conninfo=uri, open=False)
                await pool.open()
                async with pool.connection() as conn:
                    await conn.execute(CHAT_MESSAGES_DDL)
                self._pool = pool
                logger.info("Chat history: Postgres table ready")
                return
            except Exception:
                logger.exception("Chat history: Postgres unavailable; using in-memory")
        self._pool = None
        logger.warning("Chat history: in-memory (set DATABASE_URI for Postgres)")

    async def stop(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                logger.exception("Error closing chat history pool")
            self._pool = None

    async def add_messages(self, thread_id: str, username: str, messages: list[dict]) -> int:
        """Insert messages not already stored for the thread (dedupe by id).

        `messages` are serialized LangChain message dicts (as emitted in the
        SSE `done` event). Returns the number of rows added.
        """
        if not messages:
            return 0
        rows = [
            (
                thread_id,
                username,
                m.get("id") or uuid.uuid4().hex,
                m.get("type", "unknown"),
                m,
            )
            for m in messages
        ]
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chat_messages (thread_id, username, message_id, role, content) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (thread_id, message_id) DO NOTHING",
                    [(t, u, mid, role, Jsonb(msg)) for t, u, mid, role, msg in rows],
                )
                return cur.rowcount or 0
        added = 0
        known = self._memory_ids.setdefault(thread_id, set())
        bucket = self._memory.setdefault(thread_id, [])
        for _t, _u, message_id, role, message in rows:
            if message_id in known:
                continue
            known.add(message_id)
            bucket.append(
                {
                    "message_id": message_id,
                    "role": role,
                    "content": json.loads(json.dumps(message, default=str)),
                    "created_at": now_iso(),
                }
            )
            added += 1
        return added

    async def list_messages(self, thread_id: str) -> list[dict]:
        """All stored messages of a thread in chronological order.

        Returns the serialized message dicts (same shape as the SSE `done`
        event's `messages`), or an empty list when the thread has no rows.
        """
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT content FROM chat_messages WHERE thread_id = %s ORDER BY id",
                    (thread_id,),
                )
                return [row[0] for row in await cur.fetchall()]
        return [dict(row["content"]) for row in self._memory.get(thread_id, [])]


class Persistence:
    """Owns the checkpointer, store and chat history, set up at startup and closed at shutdown."""

    def __init__(self) -> None:
        self.checkpointer: Checkpointer | None = None
        self.store: BaseStore | None = None
        self.chat_history = ChatHistoryStore()
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

                await self.chat_history.start(settings.database_uri)

                self.backend_name = "postgres"
                logger.info("Persistence: Postgres (checkpointer + store + chat history) connected")
                return
            except Exception:
                logger.exception(
                    "Failed to connect to Postgres (%s); falling back to in-memory",
                    settings.database_uri,
                )
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        await self.chat_history.start(None)
        self.backend_name = "memory"
        logger.warning("Persistence: in-memory (set DATABASE_URI for Postgres)")

    async def stop(self) -> None:
        await self.chat_history.stop()
        for cm in (self._saver_cm, self._store_cm):
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error closing Postgres connection")
        self._saver_cm = self._store_cm = None


# Singleton, initialized in FastAPI lifespan.
persistence = Persistence()
