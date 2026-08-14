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
def _offline_env_fallback():
    """Offline tests run without DB connections: allow .env-style credentials.

    Production defaults to DB-only connections (missing default `llm`
    connection -> loud error); the offline suite has no Postgres, so tests
    opt back into the env fallback unless a test explicitly overrides it.
    """
    config.settings.connection_fallback_env = True
