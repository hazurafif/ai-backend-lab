"""Application settings loaded from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

# Load .env (OPENAI_API_KEY, OPENAI_BASE_URL, DEEPAGENTS_MODEL, ...) if present.
load_dotenv()

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant running inside a backend service.
You can:
- plan multi-step work and delegate subtasks to subagents (use the `task` tool)
- read/write/edit files in your workspace
- use MCP tools exposed by connected servers (they may return structured data
  that the frontend renders as interactive UI elements)
- remember things across conversations in your memory files

Be concise and direct. When you call tools, explain what you are doing in one
short line so the user can follow along in the live stream.
"""


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
    # Postgres DSN (postgresql+psycopg://user:pass@host:5432/db).
    # If unset, the app falls back to in-memory checkpointer/store (dev mode).
    database_uri: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URI") or None)

    # --- MCP servers ---
    # JSON string with server configs, or a path to a JSON file. Example:
    # {"weather": {"url": "http://localhost:8090/mcp", "transport": "streamable_http"},
    #  "cli-tool": {"command": "gofastmcp-tool", "args": ["serve"], "transport": "stdio"}}
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
    # with INTERRUPT_ON_JSON={"execute": true} for human approval.
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

    def load_mcp_servers(self) -> dict[str, dict[str, Any]]:
        if self.mcp_servers_json:
            return json.loads(self.mcp_servers_json)
        if os.path.exists(self.mcp_servers_file):
            with open(self.mcp_servers_file) as fh:
                return json.load(fh)
        return {}


settings = Settings()
