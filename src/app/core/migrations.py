"""Minimal SQL migration runner: applies `migrations/*.sql` in order.

Each file runs in its own transaction; applied versions are tracked in the
`schema_migrations` table, so re-runs are no-ops. In-memory mode (no
Postgres) skips migrations entirely — the dict-backed stores need no schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

# Repo root = 3 parents above this module (core/migrations.py -> src/app/core).
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

_SCHEMA_MIGRATIONS_DDL = """\
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _pending_migration_files(migrations_dir: Path, applied: set[str]) -> list[Path] | None:
    """Sorted pending `*.sql` files, or None when the dir is missing."""
    if not migrations_dir.is_dir():
        return None
    return sorted(p for p in migrations_dir.glob("*.sql") if p.name not in applied)


def _read_migration(path: Path) -> str:
    return path.read_text()


async def run_migrations(pool: AsyncConnectionPool, migrations_dir: Path = MIGRATIONS_DIR) -> bool:
    """Apply pending `*.sql` files from `migrations_dir` in filename order.

    Returns True when the schema is up to date (or nothing was pending), and
    False when the migrations directory is missing (callers may fall back to
    legacy inline DDL).
    """
    async with pool.connection() as conn:
        await conn.execute(_SCHEMA_MIGRATIONS_DDL)
        cursor = await conn.execute("SELECT version FROM schema_migrations")
        applied = {row[0] for row in await cursor.fetchall()}
    pending = await run_in_threadpool(_pending_migration_files, migrations_dir, applied)
    if pending is None:
        logger.warning("Migrations dir %s not found; skipping SQL migrations", migrations_dir)
        return False
    for path in pending:
        sql = await run_in_threadpool(_read_migration, path)
        async with pool.connection() as conn:
            await conn.execute(sql)
            await conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
        logger.info("Applied migration %s", path.name)
    return True
