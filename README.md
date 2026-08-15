# AI Backend — Deep Agents API

A FastAPI backend that wraps [LangChain Deep Agents](https://github.com/langchain-ai/deepagents)
into a full agent API: streaming chat (SSE + AI SDK), Postgres persistence,
MCP tool servers, skills, human-in-the-loop approval, and RAG knowledge bases
over Weaviate.

## Features

- **Deep agent runtime** — planning, subagents, virtual filesystem, skills & memory
- **Streaming chat** — typed SSE events (`POST /chat`) + Vercel AI SDK protocol (`POST /api/chat`)
- **Durable state** — Postgres checkpoints, chat history, users, skills; in-memory fallback for dev
- **MCP integration** — tool servers over `streamable_http` / `stdio` (gofastmcp-compatible), structured tool output
- **RAG knowledge bases** — upload files/folders → parse/chunk/embed → Weaviate hybrid search (BM25F + vectors), reranking, query rewrite
- **Agent configs & skills** — named profiles bundling model + prompt + skills + tools
- **HITL** — pause runs for human approval of shell/file tools, then resume
- **Auth** — JWT access + refresh tokens, `user` / `admin` roles
- **Web search** — self-hosted SearXNG tool, toggleable per request

## Quickstart

```bash
uv sync
cp .env.example .env        # set DEEPAGENTS_MODEL + API key; DATABASE_URI for Postgres
uv run uvicorn app.main:app --port 8000 --reload
```

- Health check: `curl http://127.0.0.1:8000/health`
- First start seeds an admin account: `admin` / `admin`
  (override: `DEFAULT_ADMIN_USERNAME` / `DEFAULT_ADMIN_PASSWORD`)
- Without `DATABASE_URI` everything runs in-memory (data lost on restart)

## Container

```bash
podman compose up -d --build            # app (:8000) + Postgres (:5432)
podman compose --profile extras up -d   # + searxng (:8092) + weaviate (:8093)
podman compose logs -f app              # follow the app logs
```

`docker compose` works identically. The app container runs as a non-root user
with a `/health` healthcheck and a 1 GB memory cap; the agent's `execute`
tool (when enabled via `EXECUTE_ENABLED=true`) runs *inside* the app
container (cwd `/app`), so the container is the boundary — not the host.

No container engine on macOS? `./scripts/run_podman.sh` (`start` | `stop` | `clean`)
runs the app capped at 1 GB RAM.

## Storage model (what the agent sees)

There is **one** filesystem: the per-user workspace. Every file-tool path and
`execute` command resolves to real files under `WORKSPACE_ROOT/<user_id>/`
(default `.workspace/` — a named volume in compose, so the repo stays clean
and users stay isolated). No virtual mounts.

Durability comes from the **workspace sync** (`services/workspace.py`):
before each run the user's store files (Postgres) are materialized into the
workspace dir; after the run the agent's changes are synced back.

| Workspace subdir | Store side | Sync policy |
|---|---|---|
| `memories/` | ns `(user,)` keys `/<name>` | down if missing on disk, always up |
| `uploads/` | ns `(user,)` keys `/<user>/<name>` | always down, always up |
| `skills/` | user's own skills ns (`("user","skills",<user>)`) | always down, **never** up (read-only) |
| root files (e.g. `/script.py`) | ns `(user,)` keys `/<name>` | up only (materialize as memories/) |

Rules of thumb:

- **Everything is executable**: skills are real files under `skills/` — copy
  one out (or read it) and run it directly; file tools and `execute` agree.
- **`memories/` and `uploads/` persist** via sync-up after every run (even
  on error/cancel). `skills/` is admin-owned — agent edits to it are
  discarded on the next run.
- **Not a sandbox**: `execute` can reach any absolute path; per-user
  isolation is the *default* (cwd + file-tool roots), not a boundary — keep
  HITL (`INTERRUPT_ON_JSON='{"execute": true}'`) in trusted environments.
- `EXECUTE_ENABLED=false` keeps the same backend but refuses every execute
  command (tool stays registered, returns "Execution not available").

## Configuration

Everything is env-driven — full list in `.env.example` (`src/app/core/config.py`):

| Concern | How |
|---|---|
| Chat model | `DEEPAGENTS_MODEL=<provider>:<model>` or the default `llm` connection's `extra.model` — no default, unconfigured = loud startup error |
| Postgres | `DATABASE_URI=postgresql://user:pass@host:5432/dbname` — tables/migrations apply at startup |
| MCP servers | `MCP_SERVERS_JSON` or `mcp_servers.json`; or store-managed via `/agent/tools` |
| Web search | `SEARXNG_URL` + `SEARXNG_ENABLED` (`podman compose --profile extras up -d searxng`) |
| Execute tool | `EXECUTE_ENABLED=true` — opt-in, off by default, dev/trusted only |
| HITL | `INTERRUPT_ON_JSON='{"execute": true, "edit_file": true}'` |
| KB embeddings | OpenAI (`EMBEDDINGS_MODEL`, default `text-embedding-3-small`) or **local MLX**: `./scripts/mlx_embeddings.sh` + `EMBEDDINGS_MLX_URL=http://127.0.0.1:8080/v1` (Qwen3-Embedding-0.6B on Apple Silicon, no API fees) |

**Stored connections (preferred over .env):** admin-managed provider credentials
in the DB via `GET|POST /connections` (+ `/connections/{name}`) — kinds `llm`,
`embeddings`, `mcp`, `weaviate`, `searxng`, one default per kind. The agent LLM
and KB embeddings resolve the default connection at startup; env keys are only
used when `CONNECTION_FALLBACK_ENV=true` (default: missing connection is a loud
error, never a silent `.env` read). **There is no default model**: the agent's
model comes from `DEEPAGENTS_MODEL`, or from the default `llm` connection's
`extra.model` (e.g. `"extra": {"model": "openai:deepseek-v4-flash"}`). The
app always starts — chats return a clear 503 until a model is configured
in-app (POST /connections or PUT /settings), no restart needed. Runtime
overrides: `GET|PUT /settings`.

## API

Main surface — the full endpoint table and the SSE contract live in
[./docs/api-reference.md](./docs/api-reference.md):

| Endpoint | Description |
|---|---|
| `POST /login` `/refresh` `/register` | JWT auth (access 30 min + refresh 7 days) |
| `POST /chat` | **SSE stream** of agent events (`message_delta`, `tool_*`, `subagent`, `interrupt`, `done`); multipart file uploads supported |
| `POST /api/chat` | AI SDK data-stream protocol for `useChat` frontends, incl. HITL resume |
| `GET /threads` + `/threads/{id}/...` | History, usage/cost report, LLM title + follow-up suggestions, resume, cancel, share |
| `GET\|POST /agents` + `/agents/{name}` | Named agent profiles (model + prompt + skills + tools + `thinking`) |
| `GET\|POST /skills` + `/skills/{name}` | User-scoped private skills |
| `GET /agent/skills` + `/agent/tools` (CRUD) | Global skills & MCP tool servers (admin) |
| `GET\|POST /knowledge` + `/knowledge/{id}/...` | RAG: upload files/zip, hybrid search, reindex, delete |
| `GET\|POST /connections`, `GET\|PUT /settings` | Stored provider credentials, runtime settings (admin) |
| `GET /users` + `/users/{username}` (admin) | User management |
| `GET /health` | Persistence, MCP servers, model, searxng, execute, resources |

## Feature highlights

- **Agent configs** — `{"name": "research", "model": "anthropic:claude-sonnet-4-5", "skills": [...], "tools": [...], "thinking": "xhigh"}`; resolution: your own → global → builtin `default`. Graphs are built lazily and cached; conversations survive rebuilds.
- **Skills** — skill-creator layout (`SKILL.md` + bundled files) in the durable store, **fully per-user**: `/skills` (self-service) and `/agent/skills?username=` (admin management). The default agent sees only the user's own skills; named agent configs can attach the user's skills explicitly.
- **Knowledge bases** — `POST /knowledge/{id}/files` (multipart, per-file `path` for folders) or `/zip`; per-user quota, `?alpha=` hybrid tuning, reranking (`KB_RERANK_MODEL`), query rewrite, retrieval eval via `scripts/kb_eval.py`. See [./docs/knowledge-base-plan.md](./docs/knowledge-base-plan.md) and [./docs/rag-techniques-research.md](./docs/rag-techniques-research.md).
- **HITL** — `event: interrupt` → approval UI → `POST /threads/{id}/resume` with `approve` / `edit` / `reject` / `respond`.
- **Sharing** — `POST /threads/{id}/share` → public read-only link at `/shared/{token}` (no auth).

## Project layout

```
src/app/
  main.py           FastAPI factory; normative SSE contract in the docstring
  api/v1/           versioned routers: auth, chat, agent, health, kb, connections, settings
  core/             config, database (Postgres + in-memory fallback), security, rate limiting, run registry
  services/         agent factory, chat streaming, AI SDK bridge, MCP client, skills, KB pipeline
  schema/           per-domain Pydantic v2 models
  migrations/       SQL migrations (applied at startup)
tests/              offline pytest suite — no API key, in-memory persistence
scripts/            live E2E, MCP test server, MLX embeddings, reranker + KB eval
```

## Development

```bash
uv run pytest -q                     # offline suite (no API key needed) — must be green before commits
uv run ruff check . && uv run ruff format .
uv run pre-commit install            # ruff + conventional-commits hooks
```

## Docs

- [./docs/api-reference.md](./docs/api-reference.md) — full endpoint table, SSE contract, agent configs, MCP wiring, KB tuning, HITL, sharing
- [./docs/knowledge-base-plan.md](./docs/knowledge-base-plan.md) — RAG pipeline design & rationale
- [./docs/rag-techniques-research.md](./docs/rag-techniques-research.md) — retrieval research (chunking, reranking, eval)
- [AGENTS.md](AGENTS.md) — architecture map + dev conventions (structure, async rules, git workflow)
