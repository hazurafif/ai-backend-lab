"""Persistence layer: Postgres checkpointer + store + chat history + users, in-memory fallback.

- Checkpointer (threads/conversation state) -> AsyncPostgresSaver
- Store (long-term memory, thread metadata)   -> PostgresStore
- Chat history (readable message rows)        -> chat_messages table
- Users (auth: /register, /login, /users/me)  -> users table
- User preferences (web search toggle)          -> user_preferences table
All come from `langgraph-checkpoint-postgres` / `psycopg`. When `DATABASE_URI`
is unset (no Postgres available), we fall back to `InMemorySaver` +
`InMemoryStore` + in-memory dicts so the backend still runs locally for
development.

SQL schema lives in `migrations/*.sql` (applied at startup by
`core/migrations.py`). On first start with an empty users store, a default
admin account is seeded (see `Persistence.ensure_default_admin`).
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import suppress
from typing import Any

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
        """Full user row (incl. hashed_password, role) or None when unknown."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT username, email, full_name, hashed_password, disabled, role "
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
                    "role": row[5],
                }
        return self._memory.get(username)

    async def create_user(
        self,
        username: str,
        hashed_password: str,
        email: str | None = None,
        full_name: str | None = None,
        disabled: bool = False,
        role: str = "user",
    ) -> dict | None:
        """Insert a user; returns the row, or None when the username is taken."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (username, email, full_name, hashed_password, disabled, role) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                    (username, email, full_name, hashed_password, disabled, role),
                )
                if cur.rowcount == 0:
                    return None
                return {
                    "username": username,
                    "email": email,
                    "full_name": full_name,
                    "hashed_password": hashed_password,
                    "disabled": disabled,
                    "role": role,
                }
        if username in self._memory:
            return None
        user = {
            "username": username,
            "email": email,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "disabled": disabled,
            "role": role,
        }
        self._memory[username] = user
        return user

    async def update_user(
        self,
        username: str,
        *,
        role: str | None = None,
        disabled: bool | None = None,
        hashed_password: str | None = None,
    ) -> dict | None:
        """Update role, disabled state and/or password; returns the updated row or None when unknown."""
        sets: list[str] = []
        params: list = []
        if role is not None:
            sets.append("role = %s")
            params.append(role)
        if disabled is not None:
            sets.append("disabled = %s")
            params.append(disabled)
        if hashed_password is not None:
            sets.append("hashed_password = %s")
            params.append(hashed_password)
        if not sets:
            return await self.get_user(username)
        params.append(username)
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE users SET {', '.join(sets)} WHERE username = %s", tuple(params)
                )
                if cur.rowcount == 0:
                    return None
        else:
            user = self._memory.get(username)
            if user is None:
                return None
            if role is not None:
                user["role"] = role
            if disabled is not None:
                user["disabled"] = disabled
            if hashed_password is not None:
                user["hashed_password"] = hashed_password
        return await self.get_user(username)

    async def delete_user(self, username: str) -> bool:
        """Remove a user; returns False when unknown. Their threads/history rows stay (orphaned)."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("DELETE FROM users WHERE username = %s", (username,))
                return cur.rowcount > 0
        return self._memory.pop(username, None) is not None

    async def list_users(self) -> list[dict]:
        """All users, newest first (no hashed_password)."""
        if self._use_postgres:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT username, email, full_name, disabled, role FROM users ORDER BY id DESC"
                )
                return [
                    {
                        "username": row[0],
                        "email": row[1],
                        "full_name": row[2],
                        "disabled": row[3],
                        "role": row[4],
                    }
                    for row in await cur.fetchall()
                ]
        users = list(self._memory.values())
        users.reverse()  # newest first (insertion order)
        return [
            {
                "username": u["username"],
                "email": u.get("email"),
                "full_name": u.get("full_name"),
                "disabled": u.get("disabled", False),
                "role": u.get("role", "user"),
            }
            for u in users
        ]

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

    async def replace_messages(self, thread_id: str, username: str, messages: list[dict]) -> int:
        """Rewrite the thread's rows as the full conversation, in order.

        Incremental writes (`add_messages` during streaming) land in stream
        order, which can differ from conversation order around instant tool
        calls — a completed run rewrites the thread so the table is always
        readable in conversation order. Deduped by message id.
        """
        await self.delete_thread(thread_id)
        return await self.add_messages(thread_id, username, messages)

    async def delete_thread(self, thread_id: str) -> int:
        """Remove all message rows of a thread; returns the number of rows deleted."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("DELETE FROM chat_messages WHERE thread_id = %s", (thread_id,))
                return cur.rowcount or 0
        deleted = len(self._memory.pop(thread_id, []))
        self._memory_ids.pop(thread_id, None)
        return deleted

    async def delete_threads(self, thread_ids: list[str]) -> int:
        """Remove all message rows of the given threads (bulk delete-all)."""
        if not thread_ids:
            return 0
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chat_messages WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
                return cur.rowcount or 0
        deleted = 0
        for thread_id in thread_ids:
            deleted += len(self._memory.pop(thread_id, []))
            self._memory_ids.pop(thread_id, None)
        return deleted

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


class KbStore:
    """Knowledge bases + ingested documents in Postgres (`kb`, `kb_documents`).

    Owner-scoped: every method takes `owner` so users can only touch their own
    data. Rows are plain dicts shaped like `schema/kb_schema` outputs.
    Falls back to in-memory dicts when Postgres is off (dev mode).
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._memory_kbs: dict[str, dict[str, dict]] = {}  # owner -> kb_id -> row
        self._memory_docs: dict[str, dict[str, dict]] = {}  # owner -> doc_id -> row

    @property
    def is_postgres(self) -> bool:
        return self._pool is not None

    async def start(self, pool: AsyncConnectionPool | None) -> None:
        self._pool = pool
        self._memory_kbs, self._memory_docs = {}, {}

    async def stop(self) -> None:
        self._pool = None
        self._memory_kbs, self._memory_docs = {}, {}

    # ------------------------------------------------------------------ kbs

    async def create_kb(self, owner: str, name: str, description: str | None) -> dict | None:
        """Insert a KB; returns the row, or None when (owner, name) is taken."""
        now = now_iso()
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO kb (owner, name, description) VALUES (%s, %s, %s) "
                    "ON CONFLICT (owner, name) DO NOTHING RETURNING id, created_at, updated_at",
                    (owner, name, description),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "name": name,
                    "description": description,
                    "created_at": row[1].isoformat(),
                    "updated_at": row[2].isoformat(),
                }
        bucket = self._memory_kbs.setdefault(owner, {})
        if any(kb["name"] == name for kb in bucket.values()):
            return None
        kb_id = str(uuid.uuid4())
        row = {
            "id": kb_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }
        bucket[kb_id] = row
        return dict(row)

    async def get_kb(self, owner: str, kb_id: str) -> dict | None:
        """KB metadata of `owner` incl. stats (None when unknown or not owned)."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT k.id, k.name, k.description, k.created_at, k.updated_at, "
                    "count(d.id) AS document_count, coalesce(sum(d.chunk_count), 0) AS chunk_count "
                    "FROM kb k LEFT JOIN kb_documents d ON d.kb_id = k.id "
                    "WHERE k.owner = %s AND k.id = %s GROUP BY k.id",
                    (owner, kb_id),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3].isoformat(),
                    "updated_at": row[4].isoformat(),
                    "document_count": row[5],
                    "chunk_count": row[6] or 0,
                }
        kb = self._memory_kbs.get(owner, {}).get(kb_id)
        if kb is None:
            return None
        out = dict(kb)
        docs = [d for d in self._memory_docs.get(owner, {}).values() if d["kb_id"] == kb_id]
        out["document_count"] = len(docs)
        out["chunk_count"] = sum(d["chunk_count"] for d in docs)
        return out

    async def list_kbs(self, owner: str) -> list[dict]:
        """All KBs of `owner`, newest first, with document/chunk stats."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT k.id, k.name, k.description, k.created_at, k.updated_at, "
                    "count(d.id) AS document_count, coalesce(sum(d.chunk_count), 0) AS chunk_count "
                    "FROM kb k LEFT JOIN kb_documents d ON d.kb_id = k.id "
                    "WHERE k.owner = %s GROUP BY k.id ORDER BY k.updated_at DESC",
                    (owner,),
                )
                return [
                    {
                        "id": str(r[0]),
                        "name": r[1],
                        "description": r[2],
                        "created_at": r[3].isoformat(),
                        "updated_at": r[4].isoformat(),
                        "document_count": r[5],
                        "chunk_count": r[6] or 0,
                    }
                    for r in await cur.fetchall()
                ]
        kbs = [dict(kb) for kb in self._memory_kbs.get(owner, {}).values()]
        for kb in kbs:
            docs = [d for d in self._memory_docs.get(owner, {}).values() if d["kb_id"] == kb["id"]]
            kb["document_count"] = len(docs)
            kb["chunk_count"] = sum(d["chunk_count"] for d in docs)
        kbs.sort(key=lambda k: k["updated_at"], reverse=True)
        return kbs

    async def update_kb(
        self, owner: str, kb_id: str, *, name: str | None, description: str | None
    ) -> dict | None:
        """Rename / re-describe a KB; None when unknown."""
        if self._pool is not None:
            sets, params = [], []
            if name is not None:
                sets.append("name = %s")
                params.append(name)
            if description is not None:
                sets.append("description = %s")
                params.append(description)
            if not sets:
                return await self.get_kb(owner, kb_id)
            params += [owner, kb_id]
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE kb SET {', '.join(sets)}, updated_at = now() "
                    "WHERE owner = %s AND id = %s RETURNING id, name, description, "
                    "created_at, updated_at",
                    tuple(params),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3].isoformat(),
                    "updated_at": row[4].isoformat(),
                }
        kb = self._memory_kbs.get(owner, {}).get(kb_id)
        if kb is None:
            return None
        if name is not None:
            kb["name"] = name
        if description is not None:
            kb["description"] = description
        kb["updated_at"] = now_iso()
        return dict(kb)

    async def delete_kb(self, owner: str, kb_id: str) -> bool:
        """Remove a KB and its documents (cascade). Vectors are the caller's job."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("DELETE FROM kb WHERE owner = %s AND id = %s", (owner, kb_id))
                return cur.rowcount > 0
        return self._memory_kbs.get(owner, {}).pop(kb_id, None) is not None

    async def total_bytes(self, owner: str) -> int:
        """Sum of raw document bytes across all KBs of `owner` (quota check)."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT coalesce(sum(size_bytes), 0) FROM kb_documents WHERE owner = %s",
                    (owner,),
                )
                return (await cur.fetchone())[0] or 0
        return sum(d["size_bytes"] for d in self._memory_docs.get(owner, {}).values())

    # ------------------------------------------------------------- documents

    async def add_document(
        self,
        owner: str,
        kb_id: str,
        path: str,
        mime_type: str | None,
        size_bytes: int,
        content: bytes,
    ) -> dict | None:
        """Insert a document with status `pending`; None when the KB is unknown
        or the (kb_id, path) pair already exists."""
        now = now_iso()
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO kb_documents (kb_id, owner, path, mime_type, size_bytes, content) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (kb_id, path) DO NOTHING "
                    "RETURNING id, created_at, updated_at",
                    (kb_id, owner, path, mime_type, size_bytes, content),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "kb_id": kb_id,
                    "owner": owner,
                    "path": path,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "status": "pending",
                    "error": None,
                    "chunk_count": 0,
                    "created_at": row[1].isoformat(),
                    "updated_at": row[2].isoformat(),
                }
        if kb_id not in self._memory_kbs.get(owner, {}):
            return None
        docs = self._memory_docs.setdefault(owner, {})
        if any(d["path"] == path for d in docs.values()):
            return None
        doc_id = str(uuid.uuid4())
        row = {
            "id": doc_id,
            "kb_id": kb_id,
            "owner": owner,
            "path": path,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "content": content,
            "status": "pending",
            "error": None,
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        docs[doc_id] = row
        out = dict(row)
        out.pop("content", None)
        return out

    async def get_document(self, owner: str, doc_id: str) -> dict | None:
        """Document metadata (no blob); None when unknown or not owned."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, kb_id, path, mime_type, size_bytes, status, error, "
                    "chunk_count, created_at, updated_at FROM kb_documents "
                    "WHERE owner = %s AND id = %s",
                    (owner, doc_id),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "kb_id": str(row[1]),
                    "path": row[2],
                    "mime_type": row[3],
                    "size_bytes": row[4],
                    "status": row[5],
                    "error": row[6],
                    "chunk_count": row[7],
                    "created_at": row[8].isoformat(),
                    "updated_at": row[9].isoformat(),
                }
        doc = self._memory_docs.get(owner, {}).get(doc_id)
        if doc is None:
            return None
        out = dict(doc)
        out.pop("content", None)
        return out

    async def get_document_content(self, owner: str, doc_id: str) -> tuple[dict, bytes] | None:
        """Document metadata + raw blob (for re-indexing); None when unknown."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, kb_id, path, mime_type, size_bytes, status, error, "
                    "chunk_count, created_at, updated_at, content FROM kb_documents "
                    "WHERE owner = %s AND id = %s",
                    (owner, doc_id),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                meta = {
                    "id": str(row[0]),
                    "kb_id": str(row[1]),
                    "path": row[2],
                    "mime_type": row[3],
                    "size_bytes": row[4],
                    "status": row[5],
                    "error": row[6],
                    "chunk_count": row[7],
                    "created_at": row[8].isoformat(),
                    "updated_at": row[9].isoformat(),
                }
                return meta, bytes(row[10])
        doc = self._memory_docs.get(owner, {}).get(doc_id)
        if doc is None:
            return None
        out = dict(doc)
        content = out.pop("content")
        return out, content

    async def list_documents(self, owner: str, kb_id: str) -> list[dict]:
        """All documents of a KB (metadata only), by path."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, kb_id, path, mime_type, size_bytes, status, error, "
                    "chunk_count, created_at, updated_at FROM kb_documents "
                    "WHERE owner = %s AND kb_id = %s ORDER BY path",
                    (owner, kb_id),
                )
                return [
                    {
                        "id": str(r[0]),
                        "kb_id": str(r[1]),
                        "path": r[2],
                        "mime_type": r[3],
                        "size_bytes": r[4],
                        "status": r[5],
                        "error": r[6],
                        "chunk_count": r[7],
                        "created_at": r[8].isoformat(),
                        "updated_at": r[9].isoformat(),
                    }
                    for r in await cur.fetchall()
                ]
        docs = [dict(d) for d in self._memory_docs.get(owner, {}).values() if d["kb_id"] == kb_id]
        for doc in docs:
            doc.pop("content", None)
        docs.sort(key=lambda d: d["path"])
        return docs

    async def update_document(
        self,
        owner: str,
        doc_id: str,
        *,
        status: str | None = None,
        error: str | None = None,
        clear_error: bool = False,
        chunk_count: int | None = None,
    ) -> bool:
        """Update ingest status/error/chunk_count of a document."""
        sets, params = [], []
        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if clear_error:
            sets.append("error = NULL")
        elif error is not None:
            sets.append("error = %s")
            params.append(error)
        if chunk_count is not None:
            sets.append("chunk_count = %s")
            params.append(chunk_count)
        if not sets:
            return await self.get_document(owner, doc_id) is not None
        params += [owner, doc_id]
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE kb_documents SET {', '.join(sets)}, updated_at = now() "
                    "WHERE owner = %s AND id = %s",
                    tuple(params),
                )
                return cur.rowcount > 0
        doc = self._memory_docs.get(owner, {}).get(doc_id)
        if doc is None:
            return False
        if status is not None:
            doc["status"] = status
        if clear_error:
            doc["error"] = None
        elif error is not None:
            doc["error"] = error
        if chunk_count is not None:
            doc["chunk_count"] = chunk_count
        doc["updated_at"] = now_iso()
        return True

    async def delete_document(self, owner: str, doc_id: str) -> dict | None:
        """Remove a document; returns its metadata (for vector cleanup) or None."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM kb_documents WHERE owner = %s AND id = %s "
                    "RETURNING id, kb_id, path",
                    (owner, doc_id),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {"id": str(row[0]), "kb_id": str(row[1]), "path": row[2]}
        doc = self._memory_docs.get(owner, {}).pop(doc_id, None)
        if doc is None:
            return None
        return {"id": doc["id"], "kb_id": doc["kb_id"], "path": doc["path"]}


class ConnectionStore:
    """Provider connections (base URL + API token) in Postgres (`connections` table).

    Global infra config (admin-managed, not owner-scoped): the agent LLM and
    KB embeddings resolve the default connection of their kind here instead of
    .env credentials. Rows are plain dicts shaped like `schema/connection_schema`
    outputs (with the full api_token; masking happens at the API layer).
    Falls back to in-memory dicts when Postgres is off (dev mode).
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._memory: dict[str, dict] = {}  # name -> row

    @property
    def is_postgres(self) -> bool:
        return self._pool is not None

    async def start(self, pool: AsyncConnectionPool | None) -> None:
        self._pool = pool
        self._memory = {}

    async def stop(self) -> None:
        self._pool = None
        self._memory = {}

    @staticmethod
    def _row(**kw: object) -> dict:
        return {
            "id": kw.get("id") or str(uuid.uuid4()),
            "name": kw.get("name"),
            "kind": kw.get("kind", "llm"),
            "base_url": kw.get("base_url"),
            "api_token": kw.get("api_token"),
            "extra": kw.get("extra") or {},
            "is_default": bool(kw.get("is_default", False)),
            "created_at": kw.get("created_at") or now_iso(),
            "updated_at": kw.get("updated_at") or now_iso(),
        }

    async def _clear_default(self, kind: str, keep_name: str) -> None:
        """Unset is_default for every other connection of the kind."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "UPDATE connections SET is_default = false, updated_at = now() "
                    "WHERE kind = %s AND name <> %s AND is_default",
                    (kind, keep_name),
                )
            return
        for row in self._memory.values():
            if row["kind"] == kind and row["name"] != keep_name:
                row["is_default"] = False

    async def create(self, connection: dict) -> dict | None:
        """Insert a connection; returns the row, or None when the name is taken.

        `connection` is the raw payload dict (name, kind, base_url, api_token,
        extra, is_default); when `is_default` is true it becomes the only
        default of its kind.
        """
        now = now_iso()
        if connection.get("is_default"):
            await self._clear_default(connection["kind"], connection["name"])
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO connections (name, kind, base_url, api_token, extra, is_default) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING "
                    "RETURNING id, created_at, updated_at",
                    (
                        connection["name"],
                        connection["kind"],
                        connection.get("base_url"),
                        connection.get("api_token"),
                        Jsonb(connection.get("extra") or {}),
                        connection.get("is_default", False),
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                out = self._row(
                    id=str(row[0]),
                    created_at=row[1].isoformat(),
                    updated_at=row[2].isoformat(),
                    **connection,
                )
                self._memory[connection["name"]] = out
                return out
        if connection["name"] in self._memory:
            return None
        out = self._row(**connection, id=str(uuid.uuid4()), created_at=now, updated_at=now)
        self._memory[out["name"]] = out
        return dict(out)

    async def get(self, name: str) -> dict | None:
        """Connection by name, or None when unknown."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, kind, base_url, api_token, extra, is_default, "
                    "created_at, updated_at FROM connections WHERE name = %s",
                    (name,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "name": row[1],
                    "kind": row[2],
                    "base_url": row[3],
                    "api_token": row[4],
                    "extra": row[5] or {},
                    "is_default": row[6],
                    "created_at": row[7].isoformat(),
                    "updated_at": row[8].isoformat(),
                }
        row = self._memory.get(name)
        return dict(row) if row is not None else None

    async def list(self) -> list[dict]:
        """All connections, newest first."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, kind, base_url, api_token, extra, is_default, "
                    "created_at, updated_at FROM connections ORDER BY created_at DESC"
                )
                return [
                    {
                        "id": str(r[0]),
                        "name": r[1],
                        "kind": r[2],
                        "base_url": r[3],
                        "api_token": r[4],
                        "extra": r[5] or {},
                        "is_default": r[6],
                        "created_at": r[7].isoformat(),
                        "updated_at": r[8].isoformat(),
                    }
                    for r in await cur.fetchall()
                ]
        rows = [dict(r) for r in self._memory.values()]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows

    async def update(self, name: str, patch: dict) -> dict | None:
        """Update a connection; None when unknown. `patch` may carry any subset
        of kind/base_url/api_token/extra/is_default. When is_default is set it
        becomes the only default of its kind.
        """
        existing = await self.get(name)
        if existing is None:
            return None
        if patch.get("is_default"):
            await self._clear_default(patch.get("kind", existing["kind"]), name)
        if self._pool is not None:
            sets, params = [], []
            for key, col in (
                ("kind", "kind"),
                ("base_url", "base_url"),
                ("api_token", "api_token"),
                ("is_default", "is_default"),
            ):
                if key in patch:
                    sets.append(f"{col} = %s")
                    params.append(patch[key])
            if "extra" in patch:
                sets.append("extra = %s")
                params.append(Jsonb(patch["extra"] or {}))
            if not sets:
                return existing
            params.append(name)
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE connections SET {', '.join(sets)}, updated_at = now() "
                    "WHERE name = %s RETURNING id, name, kind, base_url, api_token, "
                    "extra, is_default, created_at, updated_at",
                    tuple(params),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                out = {
                    "id": str(row[0]),
                    "name": row[1],
                    "kind": row[2],
                    "base_url": row[3],
                    "api_token": row[4],
                    "extra": row[5] or {},
                    "is_default": row[6],
                    "created_at": row[7].isoformat(),
                    "updated_at": row[8].isoformat(),
                }
                self._memory[name] = out
                return out
        for key in ("kind", "base_url", "api_token", "extra", "is_default"):
            if key in patch:
                existing[key] = patch[key]
        existing["updated_at"] = now_iso()
        return dict(existing)

    async def delete(self, name: str) -> bool:
        """Remove a connection; returns False when unknown."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("DELETE FROM connections WHERE name = %s", (name,))
                removed = cur.rowcount > 0
                if removed:
                    self._memory.pop(name, None)
                return removed
        return self._memory.pop(name, None) is not None

    async def get_default(self, kind: str) -> dict | None:
        """The default connection of a kind: is_default first, else first created."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, kind, base_url, api_token, extra, is_default, "
                    "created_at, updated_at FROM connections "
                    "WHERE kind = %s ORDER BY is_default DESC, created_at ASC LIMIT 1",
                    (kind,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "name": row[1],
                    "kind": row[2],
                    "base_url": row[3],
                    "api_token": row[4],
                    "extra": row[5] or {},
                    "is_default": row[6],
                    "created_at": row[7].isoformat(),
                    "updated_at": row[8].isoformat(),
                }
        candidates = [r for r in self._memory.values() if r["kind"] == kind]
        if not candidates:
            return None
        candidates.sort(key=lambda r: (not r["is_default"], r["created_at"]))
        return dict(candidates[0])


class AppSettingsStore:
    """Admin app settings (key-value JSON) in Postgres (`app_settings` table).

    Global infra configuration that overrides .env defaults at runtime, e.g.
    `execute` ({"enabled", "max_timeout", "inherit_env"}) and `connections`
    ({"execute"}). Falls back to in-memory dicts when Postgres is off
    (dev mode).
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._memory: dict[str, dict] = {}  # key -> value dict

    @property
    def is_postgres(self) -> bool:
        return self._pool is not None

    async def start(self, pool: AsyncConnectionPool | None) -> None:
        self._pool = pool
        self._memory = {}

    async def stop(self) -> None:
        self._pool = None
        self._memory = {}

    async def list(self) -> list[dict]:
        """All settings as {"key": ..., "value": ...} rows, newest first."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT key, value, updated_at FROM app_settings ORDER BY key")
                rows = await cur.fetchall()
            return [
                {"key": r[0], "value": dict(r[1] or {}), "updated_at": r[2].isoformat()}
                for r in rows
            ]
        return [{"key": k, "value": dict(v), "updated_at": None} for k, v in self._memory.items()]

    async def get(self, key: str) -> dict | None:
        """The stored value of a setting key, or None when unset."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
                row = await cur.fetchone()
            return dict(row[0] or {}) if row else None
        value = self._memory.get(key)
        return dict(value) if value is not None else None

    async def set(self, key: str, value: dict) -> None:
        """Upsert a setting key."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO app_settings (key, value, updated_at) VALUES (%s, %s, now()) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                    "updated_at = now()",
                    (key, Jsonb(value)),
                )
            return
        self._memory[key] = dict(value)

    async def delete(self, key: str) -> None:
        """Remove a setting key (reverts to the .env default)."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("DELETE FROM app_settings WHERE key = %s", (key,))
            return
        self._memory.pop(key, None)


class UserPreferencesStore:
    """Per-user preferences (key-value JSON) in Postgres (`user_preferences` table).

    Owner-scoped: every method takes `username` so users only read/write their
    own row. Currently carries the web search toggle (`enable_search`); the
    JSONB value keeps future preferences additive without new migrations.
    Falls back to in-memory dicts when Postgres is off (dev mode).
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._memory: dict[str, dict] = {}  # username -> {key: value}

    @property
    def is_postgres(self) -> bool:
        return self._pool is not None

    async def start(self, pool: AsyncConnectionPool | None) -> None:
        self._pool = pool
        self._memory = {}

    async def stop(self) -> None:
        self._pool = None
        self._memory = {}

    async def get(self, username: str, key: str) -> Any | None:
        """The stored value of a preference key, or None when unset."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM user_preferences WHERE username = %s", (username,)
                )
                row = await cur.fetchone()
            return (row[0] or {}).get(key) if row else None
        return self._memory.get(username, {}).get(key)

    async def get_all(self, username: str) -> dict:
        """All stored preferences of a user, as a plain dict."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM user_preferences WHERE username = %s", (username,)
                )
                row = await cur.fetchone()
            return dict(row[0] or {}) if row else {}
        return dict(self._memory.get(username, {}))

    async def set(self, username: str, key: str, value: Any) -> None:
        """Upsert a preference key; None removes the key (and the row when empty)."""
        if value is None:
            await self.delete(username, key)
            return
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO user_preferences (username, value, updated_at) "
                    "VALUES (%s, %s, now()) "
                    "ON CONFLICT (username) DO UPDATE SET "
                    "value = user_preferences.value || EXCLUDED.value, updated_at = now()",
                    (username, Jsonb({key: value})),
                )
            return
        self._memory.setdefault(username, {})[key] = value

    async def delete(self, username: str, key: str) -> None:
        """Remove a preference key; drops the row when it becomes empty."""
        if self._pool is not None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM user_preferences WHERE username = %s", (username,)
                )
                row = await cur.fetchone()
                if row is None:
                    return
                value = dict(row[0] or {})
                value.pop(key, None)
                if value:
                    await cur.execute(
                        "UPDATE user_preferences SET value = %s, updated_at = now() "
                        "WHERE username = %s",
                        (Jsonb(value), username),
                    )
                else:
                    await cur.execute(
                        "DELETE FROM user_preferences WHERE username = %s", (username,)
                    )
            return
        self._memory.get(username, {}).pop(key, None)


class Persistence:
    """Owns the checkpointer, store, chat history and users; set up at startup and closed at shutdown."""

    def __init__(self) -> None:
        self.checkpointer: Checkpointer | None = None
        self.store: BaseStore | None = None
        self.chat_history = ChatHistoryStore()
        self.users = UserStore()
        self.kb = KbStore()
        self.connections = ConnectionStore()
        self.settings = AppSettingsStore()
        self.preferences = UserPreferencesStore()
        self.backend_name = "memory"
        self._pool: AsyncConnectionPool | None = None
        self._saver_cm = None
        self._store_cm = None

    @property
    def is_postgres(self) -> bool:
        return self.backend_name == "postgres"

    async def start(self) -> None:
        pool: AsyncConnectionPool | None = None
        if settings.database_uri:
            try:
                pool = AsyncConnectionPool(conninfo=settings.database_uri, open=False)
                await pool.open()
                # Apply migrations/*.sql (users, chat_messages, ...). A missing
                # migrations dir means no schema: treat Postgres as unavailable
                # and fall back to the in-memory stores below (the legacy inline
                # chat DDL fallback is gone).
                if await run_migrations(pool):
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
                else:
                    logger.error("Migrations dir not found; falling back to in-memory stores")
                    await pool.close()
                    pool = None
            except Exception:
                logger.exception(
                    "Failed to connect to Postgres (%s); falling back to in-memory",
                    settings.database_uri,
                )
                if pool is not None:
                    with suppress(Exception):
                        await pool.close()
                pool = None

        if pool is not None:
            await self.users.start(pool, True)
            await self.chat_history.start(pool)
            await self.kb.start(pool)
            await self.connections.start(pool)
            await self.settings.start(pool)
            await self.preferences.start(pool)
        else:
            # Postgres unavailable (no DATABASE_URI, missing migrations dir, or
            # a failed connection): every store degrades to in-memory so the
            # app still boots for local development.
            await self.users.start(None, False)
            await self.chat_history.start(None)
            await self.kb.start(None)
            await self.connections.start(None)
            await self.settings.start(None)
            await self.preferences.start(None)
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
        await self.kb.stop()
        await self.connections.stop()
        await self.settings.stop()
        await self.preferences.stop()

    async def ensure_default_admin(self) -> None:
        """Guarantee an admin account exists; seed the default on a fresh store.

        Runs at startup. On a fresh database (no users) the default admin is
        seeded (username/password from `settings.default_admin_username` /
        `default_admin_password`). On existing installs the default admin
        username is promoted to the admin role when present (migration
        0003 backfill), so old databases never end up admin-less.
        """
        username = settings.default_admin_username
        existing = await self.users.get_user(username)
        if existing is not None:
            if existing.get("role") != "admin":
                await self.users.update_user(username, role="admin")
                logger.info("Promoted %r to admin role (default admin account)", username)
            return
        if await self.users.count() > 0:
            logger.warning(
                "No admin user exists: %r not found. Set DEFAULT_ADMIN_USERNAME "
                "or promote a user via PATCH /users/{username}.",
                username,
            )
            return
        user = await self.users.create_user(
            username=username,
            hashed_password=get_password_hash(settings.default_admin_password),
            full_name="Admin",
            role="admin",
        )
        if user is not None:
            logger.warning(
                "Seeded default admin user %r (set DEFAULT_ADMIN_USERNAME / "
                "DEFAULT_ADMIN_PASSWORD to override)",
                username,
            )


# Singleton, initialized in FastAPI lifespan.
persistence = Persistence()
