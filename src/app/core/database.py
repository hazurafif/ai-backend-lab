"""Persistence layer: Postgres checkpointer + store + chat history + users, in-memory fallback.

- Checkpointer (threads/conversation state) -> AsyncPostgresSaver
- Store (long-term memory, thread metadata)   -> PostgresStore
- Chat history (readable message rows)        -> chat_messages table
- Users (auth: /register, /login, /users/me)  -> users table
All come from `langgraph-checkpoint-postgres` / `psycopg`. When `DATABASE_URI`
is unset (no Postgres available), we fall back to `InMemorySaver` +
`InMemoryStore` + in-memory dicts so the backend still runs locally for
development.

SQL schema lives in `migrations/*.sql` (applied at startup by
`core/migrations.py`); `CHAT_MESSAGES_DDL` below is a legacy fallback used
only when the migrations folder is missing. On first start with an empty
users store, a default admin account is seeded (see
`Persistence.ensure_default_admin`).
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import suppress

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
from .migrations import run_migrations
from .security import get_password_hash

logger = logging.getLogger(__name__)


# Legacy fallback for `chat_messages` when the migrations folder is missing
# (mirrors migrations/0002_create_chat_messages.sql).
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


class UserStore:
    """Auth users in Postgres (`users` table) with an in-memory fallback.

    Rows are plain dicts shaped like `schema/auth_schema.UserInDB`
    (username, email, full_name, hashed_password, disabled). Postgres mode is
    enabled only when the SQL migrations ran; otherwise we degrade to a dict
    so the app still boots without the schema (dev mode).
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._use_postgres = False
        self._memory: dict[str, dict] = {}

    @property
    def is_postgres(self) -> bool:
        return self._use_postgres

    async def start(self, pool: AsyncConnectionPool | None, postgres_ready: bool) -> None:
        """Bind the shared pool (Postgres) or reset the in-memory fallback."""
        self._pool = pool
        self._use_postgres = postgres_ready
        self._memory = {}
        if postgres_ready:
            logger.info("Users: Postgres table ready")
        else:
            logger.warning("Users: in-memory (set DATABASE_URI for Postgres)")

    async def stop(self) -> None:
        self._pool = None
        self._use_postgres = False
        self._memory = {}

    async def get_user(self, username: str) -> dict | None:
        """Full user row (incl. hashed_password) or None when unknown."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT username, email, full_name, hashed_password, disabled "
                    "FROM users WHERE username = %s",
                    (username,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "username": row[0],
                    "email": row[1],
                    "full_name": row[2],
                    "hashed_password": row[3],
                    "disabled": row[4],
                }
        return self._memory.get(username)

    async def create_user(
        self,
        username: str,
        hashed_password: str,
        email: str | None = None,
        full_name: str | None = None,
        disabled: bool = False,
    ) -> dict | None:
        """Insert a user; returns the row, or None when the username is taken."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (username, email, full_name, hashed_password, disabled) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                    (username, email, full_name, hashed_password, disabled),
                )
                if cur.rowcount == 0:
                    return None
                return {
                    "username": username,
                    "email": email,
                    "full_name": full_name,
                    "hashed_password": hashed_password,
                    "disabled": disabled,
                }
        if username in self._memory:
            return None
        user = {
            "username": username,
            "email": email,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "disabled": disabled,
        }
        self._memory[username] = user
        return user

    async def count(self) -> int:
        """Number of registered users."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM users")
                return (await cur.fetchone())[0]
        return len(self._memory)


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

    async def start(self, pool: AsyncConnectionPool | None) -> None:
        """Bind the shared Postgres pool, or reset the in-memory fallback."""
        self._memory, self._memory_ids = {}, {}
        self._pool = pool
        if pool is not None:
            logger.info("Chat history: Postgres table ready")
        else:
            logger.warning("Chat history: in-memory (set DATABASE_URI for Postgres)")

    async def stop(self) -> None:
        self._pool = None
        self._memory, self._memory_ids = {}, {}

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
    """Owns the checkpointer, store, chat history and users; set up at startup and closed at shutdown."""

    def __init__(self) -> None:
        self.checkpointer: Checkpointer | None = None
        self.store: BaseStore | None = None
        self.chat_history = ChatHistoryStore()
        self.users = UserStore()
        self.backend_name = "memory"
        self._pool: AsyncConnectionPool | None = None
        self._saver_cm = None
        self._store_cm = None

    @property
    def is_postgres(self) -> bool:
        return self.backend_name == "postgres"

    async def start(self) -> None:
        pool: AsyncConnectionPool | None = None
        postgres_ready = False
        if settings.database_uri:
            try:
                pool = AsyncConnectionPool(conninfo=settings.database_uri, open=False)
                await pool.open()
                # Apply migrations/*.sql (users, chat_messages, ...).
                if await run_migrations(pool):
                    postgres_ready = True
                else:
                    # No migrations folder: keep the legacy inline chat DDL.
                    async with pool.connection() as conn:
                        await conn.execute(CHAT_MESSAGES_DDL)

                # AsyncPostgresSaver / PostgresStore are async context managers.
                self._saver_cm = AsyncPostgresSaver.from_conn_string(settings.database_uri)
                self.checkpointer = await self._saver_cm.__aenter__()
                await self.checkpointer.setup()  # create tables

                self._store_cm = AsyncPostgresStore.from_conn_string(settings.database_uri)
                self.store = await self._store_cm.__aenter__()
                await self.store.setup()

                self.backend_name = "postgres"
                logger.info(
                    "Persistence: Postgres (checkpointer + store + chat history + users) connected"
                )
            except Exception:
                logger.exception(
                    "Failed to connect to Postgres (%s); falling back to in-memory",
                    settings.database_uri,
                )
                if pool is not None:
                    with suppress(Exception):
                        await pool.close()
                pool = None

        await self.users.start(pool, postgres_ready)
        await self.chat_history.start(pool)
        if pool is None:
            self.checkpointer = InMemorySaver()
            self.store = InMemoryStore()
            self.backend_name = "memory"
            logger.warning("Persistence: in-memory (set DATABASE_URI for Postgres)")
        self._pool = pool
        await self.ensure_default_admin()

    async def stop(self) -> None:
        for cm in (self._saver_cm, self._store_cm):
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error closing Postgres connection")
        self._saver_cm = self._store_cm = None
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                logger.exception("Error closing persistence pool")
            self._pool = None
        await self.users.stop()
        await self.chat_history.stop()

    async def ensure_default_admin(self) -> None:
        """Seed the default admin account when the users store is empty.

        Runs at startup (first server start with a fresh database). Username
        and password come from `settings.default_admin_username` /
        `default_admin_password` (env: DEFAULT_ADMIN_USERNAME/PASSWORD).
        """
        if await self.users.count() > 0:
            return
        username = settings.default_admin_username
        user = await self.users.create_user(
            username=username,
            hashed_password=get_password_hash(settings.default_admin_password),
            full_name="Admin",
        )
        if user is not None:
            logger.warning(
                "Seeded default admin user %r (set DEFAULT_ADMIN_USERNAME / "
                "DEFAULT_ADMIN_PASSWORD to override)",
                username,
            )


# Singleton, initialized in FastAPI lifespan.
persistence = Persistence()
