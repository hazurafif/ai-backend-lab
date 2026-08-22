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

## Container (OrbStack)

```bash
docker compose up -d --build            # app (:8000) + Postgres (:5432)
docker compose --profile extras up -d   # + searxng (:8092) + weaviate (:8093)
docker compose logs -f app              # follow the app logs
```

On macOS the `docker` CLI is provided by [OrbStack](https://orbstack.dev)
(free, no Docker Desktop needed). The app container runs as a non-root user
with a `/health` healthcheck and a 1 GB memory cap; the agent's `execute`
tool (when enabled via `EXECUTE_ENABLED=true`) runs *inside* the app
container (cwd `/app`), so the container is the boundary — not the host.

Prefer plain `docker run` over compose? `./scripts/run_orbstack.sh`
(`start` | `stop` | `clean`) builds and runs the same stack, app capped at
1 GB RAM.

## Storage model (what the agent sees)

Simple by design: **a system prompt + the host file system.** There is one
filesystem — the per-user workspace. Every file-tool path and `execute`
command resolves to real files under `WORKSPACE_ROOT/<user_id>/` (default
`.workspace/`, bind-mounted into the container in compose — so
`.workspace/<username>/` shows up on the host right after user creation).
No virtual mounts, no store mirroring: the files on disk are the source of
truth.

On user creation `ensure_user_workspace` scaffolds the working dirs (each
git-tracked):

| Workspace subdir | Purpose |
|---|---|
| `memories/` | durable notes the agent keeps across conversations |
| `skills/` | the user's own skills (`/skills` API), materialized before each run — read-only, agent edits are overwritten |
| `uploads/` | files the user uploads in chat (`services/uploads.py`) |
| `tmp/` | scratch space for the agent's scripts and intermediate files |
| *(anything else)* | agent-created dirs/files, e.g. `scripts/`, project code |

The agent knows this layout **only from its system prompt** (rendered per
user, `{{username}}`) and from the per-user backend it's built with — each
agent's file tools + shell are rooted in its own dir, so it never sees
other users' workspaces.

The workspace root is **its own git repo**, initialized at startup and
auto-committed after every run. Git credentials (`GIT_TOKEN` +
`GIT_REMOTE_URL`) are written to `.git-credentials` (gitignored) at startup,
so the agent can `git push` from inside the container whenever it judges it
useful.

Rules of thumb:

- **Everything is executable**: skills are real files under `skills/` — copy
  one out (or read it) and run it directly; file tools and `execute` agree.
- **`memories/`, `uploads/` and `tmp/` persist** on disk (auto-committed
  after every run, even on error/cancel). `skills/` is user/admin-owned —
  agent edits to it are discarded on the next run.
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
| Web search | `SEARXNG_URL` + `SEARXNG_ENABLED` (`docker compose --profile extras up -d searxng`) |
| Execute tool | `EXECUTE_ENABLED=true` — opt-in, off by default, dev/trusted only |
| HITL | `INTERRUPT_ON_JSON='{"execute": true, "edit_file": true}'` |
| Logging | `LOG_LEVEL` (default `INFO`) — one console line per request with user + timing via `RequestLogMiddleware`; every log record carries `request_id`/`user_id` (echo `X-Request-ID`) |
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
