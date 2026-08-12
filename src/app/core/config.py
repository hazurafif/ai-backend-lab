"""Application settings loaded from environment variables.

Follows the production-template pattern: a module-level `settings` singleton
consumed everywhere (no scattered `os.environ` reads). Values come from the
`.env` file (or the process environment), prefixed per section in the
variable names themselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from .constants import DEFAULT_SYSTEM_PROMPT

# Load .env (OPENAI_API_KEY, OPENAI_BASE_URL, DEEPAGENTS_MODEL, ...) if present.
load_dotenv()


def _load_json_env(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


@dataclass
class Settings:
    # --- Agent ---
    # Model string, e.g. "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5", "google_genai:gemini-2.5-flash"
    # For a custom OpenAI-compatible endpoint (e.g. opencode.ai/zen/go/v1),
    # set OPENAI_BASE_URL + OPENAI_API_KEY in .env; langchain-openai reads them.
    model: str = field(
        default_factory=lambda: os.environ.get("DEEPAGENTS_MODEL", "openai:gpt-4o-mini")
    )
    system_prompt: str = field(
        default_factory=lambda: os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    )
    # e.g. {"edit_file": true} -> pause for human approval before edits
    interrupt_on: dict[str, Any] = field(
        default_factory=lambda: _load_json_env("INTERRUPT_ON_JSON", {})
    )

    # --- Persistence ---
    # Postgres DSN, e.g. postgresql://user:pass@host:5432/db (psycopg conninfo
    # or URL — NOT the SQLAlchemy-style postgresql+psycopg:// prefix).
    # If unset, the app falls back to in-memory checkpointer/store (dev mode).
    database_uri: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URI") or None)

    # --- MCP servers ---
    # JSON string with server configs, or a path to a JSON file. Example:
    # {"weather": {"url": "http://localhost:8090/mcp", "transport": "streamable_http"},
    #  "cli-tool": {"command": "gofastmcp-tool", "args": ["serve"], "transport": "stdio"}}
    # Used only when the store has no tool servers (see services/resources.py).
    mcp_servers_json: str | None = field(
        default_factory=lambda: os.environ.get("MCP_SERVERS_JSON") or None
    )
    mcp_servers_file: str = field(
        default_factory=lambda: os.environ.get("MCP_SERVERS_FILE", "mcp_servers.json")
    )

    # --- Web search (SearXNG, self-hosted) ---
    # SEARXNG_URL unset -> the web_search tool is not registered at all.
    # SEARXNG_ENABLED=false -> tool exists but returns a "disabled" message.
    # Per-request override: {"enable_search": false} in /chat and /api/chat bodies.
    searxng_url: str | None = field(default_factory=lambda: os.environ.get("SEARXNG_URL") or None)
    searxng_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEARXNG_ENABLED", "true").lower() == "true"
    )
    searxng_max_results: int = field(
        default_factory=lambda: int(os.environ.get("SEARXNG_MAX_RESULTS", "5"))
    )
    searxng_timeout: float = field(
        default_factory=lambda: float(os.environ.get("SEARXNG_TIMEOUT", "10"))
    )

    # --- Shell execution (execute tool) ---
    # Opt-in: EXECUTE_ENABLED=true swaps the default StateBackend for a
    # LocalShellBackend so the built-in `execute` tool runs host shell
    # commands (unrestricted — dev/trusted environments only).
    # Tool-level permissions for execute are not supported upstream; pair it
    # with INTERRUPT_ON_JSON={\"execute\": true} for human approval.
    execute_enabled: bool = field(
        default_factory=lambda: os.environ.get("EXECUTE_ENABLED", "false").lower() == "true"
    )
    execute_max_timeout: int = field(
        default_factory=lambda: int(os.environ.get("EXECUTE_MAX_TIMEOUT", "3600"))
    )
    execute_inherit_env: bool = field(
        default_factory=lambda: os.environ.get("EXECUTE_INHERIT_ENV", "false").lower() == "true"
    )

    # --- API ---
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
        ]
    )
    secret_key: str = field(
        default_factory=lambda: os.environ.get("SECRET_KEY", "dev-secret-change-me")
    )

    # --- Rate limiting (login brute-force protection, in-memory) ---
    # Failed logins per (client IP + username) within the window; past the
    # cap the endpoint returns 429 until failures age out.
    login_rate_limit_max: int = field(
        default_factory=lambda: int(os.environ.get("LOGIN_RATE_LIMIT_MAX", "10"))
    )
    login_rate_limit_window: float = field(
        default_factory=lambda: float(os.environ.get("LOGIN_RATE_LIMIT_WINDOW", "900"))
    )

    # --- Default admin (seeded on first start when the users store is empty) ---
    default_admin_username: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
    )
    default_admin_password: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin")
    )

    def load_mcp_servers(self) -> dict[str, dict[str, Any]]:
        if self.mcp_servers_json:
            return json.loads(self.mcp_servers_json)
        if os.path.exists(self.mcp_servers_file):
            with open(self.mcp_servers_file) as fh:
                return json.load(fh)
        return {}


settings = Settings()
