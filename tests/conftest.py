"""Shared offline-test fixtures and dummy credentials.

The password literals below are test-only fixtures and never used outside the
offline suite. The `# gitguardian:ignore` comments keep GitGuardian's
generic-password detector from flagging them as leaked secrets — do not remove
them unless the values change.
"""

TEST_PASSWORD = "super-secret"  # gitguardian:ignore
TEST_NEW_PASSWORD = "new-secret-1"  # gitguardian:ignore
