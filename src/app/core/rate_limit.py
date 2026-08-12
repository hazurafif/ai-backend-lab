"""In-memory sliding-window rate limiter for login failures.

Keeps a per-key list of failure timestamps within the window; a key is
allowed while its failure count stays below the cap. In-memory only (single
process) — adequate as a brute-force speed bump for /login, not a
distributed defense.

The module-level `login_limiter` is wired to settings values at import;
`clear()` exists so tests can reset state.
"""

from __future__ import annotations

import time

from .config import settings


class SlidingWindowRateLimiter:
    """Sliding-window limiter: `check` returns whether a request is allowed."""

    def __init__(self, max_attempts: int, window_seconds: float) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._failures: dict[str, list[float]] = {}

    def _prune(self, key: str, now: float) -> list[float]:
        cutoff = now - self.window_seconds
        kept = [t for t in self._failures.get(key, []) if t > cutoff]
        self._failures[key] = kept
        return kept

    def check(self, key: str) -> bool:
        """True when the key may try again (failure count below the cap)."""
        if self.max_attempts <= 0:
            return True
        return len(self._prune(key, time.monotonic())) < self.max_attempts

    def record_failure(self, key: str) -> None:
        """Record one failed attempt for the key."""
        self._prune(key, time.monotonic()).append(time.monotonic())

    def record_success(self, key: str) -> None:
        """Clear accumulated failures (successful login)."""
        self._failures.pop(key, None)

    def clear(self) -> None:
        self._failures.clear()


login_limiter = SlidingWindowRateLimiter(
    max_attempts=settings.login_rate_limit_max,
    window_seconds=settings.login_rate_limit_window,
)
