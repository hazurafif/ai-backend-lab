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


def _load_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class Settings:
    # --- Agent ---
    # Model string, e.g. "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5", "google_genai:gemini-2.5-flash".
    # Optional: when unset, the default `llm` connection's `extra.model` is
    # used; when neither exists chats return 503 with setup instructions
    # (the app always starts; the model is configured in-app later). For a custom OpenAI-compatible endpoint (e.g.
    # opencode.ai/zen/go/v1), the connection carries the base URL + key and
    # langchain-openai reads them.
    model: str | None = field(default_factory=lambda: os.environ.get("DEEPAGENTS_MODEL") or None)
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
    database_uri: str | None = field(
        default_factory=lambda: (
            os.environ.get("DATABASE_URI")
            or os.environ.get("DATABASE_URL")  # set by `fly postgres attach`
            or None
        )
    )

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
    # Per-call timeout (seconds) for the MCP tools proxy (POST /mcp/tools/call).
    mcp_tool_call_timeout: float = field(
        default_factory=lambda: float(os.environ.get("MCP_TOOL_CALL_TIMEOUT", "60"))
    )

    # --- Web search (SearXNG, self-hosted) ---
    # SEARXNG_URL unset -> the web_search tool is not registered at all.
    # SEARXNG_ENABLED=false -> tool exists but returns a "disabled" message.
    # Override chain: {"enable_search": bool} in /chat and /api/chat bodies,
    # else the per-user stored preference (PATCH /users/me/preferences), else
    # this config value.
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
    # When false (default), missing DB connections are an error instead of a
    # silent .env fallback for the agent LLM / KB embeddings. DB settings
    # (`app_settings.connections.fallback_env`) override this at runtime.
    connection_fallback_env: bool = field(
        default_factory=lambda: os.environ.get("CONNECTION_FALLBACK_ENV", "false").lower() == "true"
    )

    # --- Chat file uploads ---
    # Uploads are stored as real files in the user's workspace dir (uploads/),
    # versioned by the workspace git repo — no Postgres involvement.
    max_upload_size_mb: int = field(
        default_factory=lambda: int(os.environ.get("MAX_UPLOAD_SIZE_MB", "25"))
    )
    # Root of the per-user workspace (the agent's only filesystem): real
    # files under <WORKSPACE_ROOT>/<user_id>/, bind-mounted from the host's
    # ./.workspace in compose (so user dirs show up on the host). The app
    # repo stays clean; the files on disk are the source of truth.
    workspace_root: str = field(
        default_factory=lambda: os.environ.get("WORKSPACE_ROOT", ".workspace")
    )

    # --- Workspace git (versioning only; the files on disk are the source of truth) ---
    # The workspace root is its own git repo, initialized at startup and
    # auto-committed after each run. Whether to PUSH is the agent's call (via
    # the execute tool) — these settings only configure identity + credentials
    # so a push can succeed from inside the container: GIT_TOKEN +
    # GIT_REMOTE_URL are written to .git-credentials (gitignored) at startup.
    git_username: str = field(default_factory=lambda: os.environ.get("GIT_USERNAME", "ai-backend"))
    git_email: str = field(default_factory=lambda: os.environ.get("GIT_EMAIL", "ai-backend@local"))
    git_token: str | None = field(default_factory=lambda: os.environ.get("GIT_TOKEN") or None)
    git_remote_url: str | None = field(
        default_factory=lambda: os.environ.get("GIT_REMOTE_URL") or None
    )

    # --- Knowledge base (RAG) ---
    # WEAVIATE_URL unset -> no vector store: KB uploads are rejected with 503
    # and the agent's search_knowledge_base tool is not registered.
    weaviate_url: str | None = field(default_factory=lambda: os.environ.get("WEAVIATE_URL") or None)
    weaviate_api_key: str | None = field(
        default_factory=lambda: os.environ.get("WEAVIATE_API_KEY") or None
    )
    # Embeddings: OpenAIEmbeddings when OPENAI_API_KEY is present (model + optional
    # custom base URL), otherwise a deterministic local embedder (dev/tests only).
    embeddings_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
    )
    embeddings_base_url: str | None = field(
        default_factory=lambda: os.environ.get("EMBEDDINGS_BASE_URL") or None
    )
    # Matryoshka truncation for text-embedding-3 models (e.g. 1024): 3x less
    # storage/search cost for a few recall points. None = model default dims.
    embeddings_dimensions: int | None = field(
        default_factory=lambda: _load_optional_int("EMBEDDINGS_DIMENSIONS")
    )
    # Local embeddings on Apple Silicon via MLX: when EMBEDDINGS_MLX_URL is set
    # and no 'embeddings' connection exists, build_embeddings() targets this
    # OpenAI-compatible endpoint (mlx_lm.server --embedding-model, see
    # scripts/mlx_embeddings.sh) with Qwen3-Embedding-0.6B (1024 dims, Apache-2.0).
    embeddings_mlx_url: str | None = field(
        default_factory=lambda: os.environ.get("EMBEDDINGS_MLX_URL") or None
    )
    embeddings_mlx_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDINGS_MLX_MODEL", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        )
    )
    # Qwen3-Embedding is instruction-aware: prepending an instruction to
    # *queries* (not passages) gains ~1-5% retrieval quality. Empty string
    # disables; unset uses the Qwen retrieval instruction for qwen3-embedding
    # models and nothing for others.
    embeddings_query_instruction: str | None = field(
        default_factory=lambda: os.environ.get("EMBEDDINGS_QUERY_INSTRUCTION")
    )
    # BM25F field weights for Weaviate hybrid search: {"path": 2.0, "content": 1.0}
    # boosts titles/paths over body text (keyword stage only).
    kb_bm25_property_weights: dict[str, float] = field(
        default_factory=lambda: _load_json_env("KB_BM25_PROPERTY_WEIGHTS", {"path": 2.0})
    )
    # Reranking (R3): retrieve broad -> cross-encoder rerank -> top-k.
    # KB_RERANK_MODEL unset -> no reranking (identity). Local CPU option:
    # flashrank ms-marco-MiniLM-L-12-v2 (tiny, ~30ms per 20 candidates).
    kb_rerank_model: str | None = field(
        default_factory=lambda: os.environ.get("KB_RERANK_MODEL") or None
    )
    kb_rerank_candidates: int = field(
        default_factory=lambda: int(os.environ.get("KB_RERANK_CANDIDATES", "20"))
    )
    # Query rewriting (R4): LLM call before retrieval for vague queries.
    # Opt-in (KB_QUERY_REWRITE=true); failures/trivial queries degrade to the
    # original query. Model defaults to DEEPAGENTS_MODEL.
    kb_query_rewrite: bool = field(
        default_factory=lambda: os.environ.get("KB_QUERY_REWRITE", "false").lower() == "true"
    )
    kb_rewrite_model: str | None = field(
        default_factory=lambda: os.environ.get("KB_REWRITE_MODEL") or None
    )
    # Queries shorter than this are left untouched (keyword search handles them).
    kb_rewrite_min_length: int = field(
        default_factory=lambda: int(os.environ.get("KB_REWRITE_MIN_LENGTH", "8"))
    )
    kb_max_file_size_mb: int = field(
        default_factory=lambda: int(os.environ.get("KB_MAX_FILE_SIZE_MB", "25"))
    )
    kb_allowed_extensions: list[str] = field(
        default_factory=lambda: [
            e.strip()
            for e in os.environ.get(
                "KB_ALLOWED_EXTENSIONS",
                ".md,.txt,.pdf,.docx,.csv,.html,.json,.py,.js,.ts,.go,.rs,.java,.sql,.yml,.yaml,.xml",
            ).split(",")
            if e.strip()
        ]
    )
    kb_chunk_size: int = field(default_factory=lambda: int(os.environ.get("KB_CHUNK_SIZE", "1000")))
    kb_chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("KB_CHUNK_OVERLAP", "200"))
    )
    kb_max_upload_batch: int = field(
        default_factory=lambda: int(os.environ.get("KB_MAX_UPLOAD_BATCH", "100"))
    )
    # Vector vs keyword weight for Weaviate hybrid search (0 = BM25F only, 1 = vectors only).
    kb_hybrid_alpha: float = field(
        default_factory=lambda: float(os.environ.get("KB_HYBRID_ALPHA", "0.5"))
    )
    # Per-user storage quota (raw bytes of all documents across KBs).
    kb_quota_mb: int = field(default_factory=lambda: int(os.environ.get("KB_QUOTA_MB", "500")))
    # Zip upload hardening: max entries + max total uncompressed size (zip-bomb guard).
    kb_zip_max_entries: int = field(
        default_factory=lambda: int(os.environ.get("KB_ZIP_MAX_ENTRIES", "500"))
    )
    kb_zip_max_total_mb: int = field(
        default_factory=lambda: int(os.environ.get("KB_ZIP_MAX_TOTAL_MB", "100"))
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
