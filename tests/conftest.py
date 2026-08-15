"""Shared offline-test fixtures and dummy credentials.

The password literals below are test-only fixtures and never used outside the
offline suite. The `# gitguardian:ignore` comments keep GitGuardian's
generic-password detector from flagging them as leaked secrets — do not remove
them unless the values change.
"""

import pytest

from app.core import config

TEST_PASSWORD = "super-secret"  # gitguardian:ignore
TEST_NEW_PASSWORD = "new-secret-1"  # gitguardian:ignore


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path):
    """Every test gets a throwaway workspace root.

    Tests that exercise user create / chat runs would otherwise write user
    dirs and git commits into the developer's real `.workspace` (and its
    git repo). Redirecting the root for the whole suite keeps the real
    workspace clean; tests needing a specific layout override
    `settings.workspace_root` themselves (monkeypatch wins over this). The
    root is NOT restored to `.workspace` afterwards, so background run
    tasks that outlive their test still land in a temp dir, never the real
    one.
    """
    config.settings.workspace_root = str(tmp_path / "workspace")
    yield


@pytest.fixture(autouse=True)
def _offline_env_fallback():
    """Offline tests run without DB connections: allow .env-style credentials.

    Production defaults to DB-only connections (missing default `llm`
    connection -> loud error); the offline suite has no Postgres, so tests
    opt back into the env fallback unless a test explicitly overrides it.
    """
    config.settings.connection_fallback_env = True
