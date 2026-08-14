# AI Backend — Deep Agents + Postgres + MCP (gofastmcp)

A FastAPI backend wrapping **LangChain Deep Agents** as the core agent:

- **Agent**: [`deepagents`](https://github.com/langchain-ai/deepagents) — planning,
  subagents (`task` tool), virtual filesystem, context management, skills & memory.
- **Persistence**: Postgres via `langgraph-checkpoint-postgres` —
  `AsyncPostgresSaver` (conversations) + `AsyncPostgresStore` (long-term memory,
  thread metadata) + a `chat_messages` table (readable chat history). Falls back
  to in-memory when `DATABASE_URI` is unset.
- **MCP**: connects to MCP servers (e.g. built with **gofastmcp**) over
  `streamable_http` or `stdio` via `langchain-mcp-adapters`. MCP tool outputs are
  streamed to the frontend as structured events — ready to render as interactive
  UI elements (cards, charts, forms) later.
- **Web search**: a `web_search` tool backed by a self-hosted **SearXNG**
  metasearch instance, toggleable from config and per-request (frontend).
- **Shell execution**: the built-in `execute` tool (opt-in, **off by default**)
  runs host shell commands for dev/trusted environments.
- **Agent resources API**: CRUD for skills and MCP tool servers, persisted in
  the LangGraph store (Postgres) and live-wired into the agent.
- **Streaming**: `POST /chat` returns a Server-Sent Events (SSE) stream with typed
  events (message deltas, tool calls, subagents, final state).
- **Auth**: JWT login (`/login`), protected chat/thread endpoints.

## Quickstart

```bash
uv sync
cp .env.example .env      # set DEEPAGENTS_MODEL + key; DATABASE_URI for Postgres
uv run uvicorn app.main:app --port 8000 --reload
```

Dev tooling (see `AGENTS.md` for full conventions):

```bash
uv run pytest -q                    # offline tests (no API key needed)
uv run ruff check . && uv run ruff format .
uv run pre-commit install           # ruff + conventional-commits hooks
```

> A `.env` already exists locally with `OPENAI_BASE_URL=https://opencode.ai/zen/go/v1`
> and `DEEPAGENTS_MODEL=openai:deepseek-v4-flash` for testing. `.env` is gitignored.

Health check: `curl http://127.0.0.1:8000/health`

## Container

- `docker compose up -d --build` — builds and runs the app (:8000) together
  with Postgres (:5432); the app reaches the DB via the compose service
  name, and `.env` is picked up if present.
- No Docker on macOS? Podman works: `./scripts/run_podman.sh` builds the
  image, creates a pod (postgres + app) and runs the **app capped at 1 GB
  RAM** (`--memory=1g --memory-swap=1g`; the podman machine VM gets 2 GB).
  Subcommands: `start` (default), `stop`, `clean`.

## Model configuration

`DEEPAGENTS_MODEL` accepts any `provider:model` string understood by
[`init_chat_model`](https://docs.langchain.com/oss/python/langchain/models)
(`openai:...`, `anthropic:...`, `google_genai:...`, ...). For a custom
OpenAI-compatible gateway set `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `.env`
(langchain-openai reads them) — e.g. the opencode.ai zen gateway.

**Stored API connections (no `.env` keys):** admin-managed connections
(`base URL` + `api key`) live in the DB and drive the agent's chat model
instead of `OPENAI_BASE_URL`/`OPENAI_API_KEY`:

```bash
# admin: create a connection used by the builtin `default` agent
curl -X POST http://127.0.0.1:8000/agent/connections \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name": "default", "base_url": "https://opencode.ai/zen/go/v1", "api_key": "..."}'
```

- `GET /agent/connections` — list (any user; API keys are never returned)
- `POST /agent/connections` — create (admin)
- `GET|PUT|DELETE /agent/connections/{name}` — read / merge-update / delete
  (admin for writes; omitting `api_key` on update keeps the stored key)
- Agent configs reference a connection via the `connection` field
  (`POST /agents`), e.g. `{"model": "openai:gpt-4o-mini", "connection": "openai"}`.
- The connection named `default` is used by the builtin `default` agent;
  without it the env-based behavior stays.
- Connections target OpenAI-compatible endpoints (langchain-openai).

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /login` | – | OAuth2 form `username`/`password` → JWT (access + refresh token) |
| `POST /refresh` | – | Exchange a refresh token for a new access token |
| `POST /register` | – | Self-service registration (always creates a `user` role) |
| `POST /chat` | Bearer | Run the agent; **SSE stream** of events. Optional `agent` field selects a custom agent config |
| `POST /api/chat` | optional Bearer | AI SDK data-stream protocol for the frontend (`useChat`), incl. HITL resume. Optional `agent` field |
| `GET /threads` | Bearer | Conversations of the current user (newest first, `limit`/`offset` pagination; each thread carries the `agent` it runs on) |
| `GET /threads/{id}/messages` | Bearer | Full history of a thread |
| `PATCH /threads/{id}` | Bearer | Rename a thread |
| `DELETE /threads/{id}` | Bearer | Delete a thread (state + history + metadata) |
| `POST /threads/{id}/resume` | Bearer | Resume a run paused for human approval |
| `POST /threads/{id}/cancel` | Bearer | Abort the active run of a thread (`done` event carries `cancelled: true`) |
| `POST /threads/{id}/share` | Bearer | Create a public share link (owner; idempotent) |
| `GET /threads/{id}/share` | Bearer | Current share link of a thread (owner) |
| `DELETE /threads/{id}/share` | Bearer | Revoke the thread's share link (owner) |
| `GET /shared/{token}` | – | **Public** read-only view of a shared thread (no auth) |
| `GET /agent/skills` + CRUD | read: any user; write: Bearer (admin) | Manage **global** skills (SKILL.md + bundled files, applied on next run) |
| `DELETE /agent/skills/{name}/files/{path}` | Bearer (admin) | Delete one bundled global skill file |
| `GET/POST /skills` + `GET/PUT/DELETE /skills/{name}` | Bearer | **User-scoped** "my skills" (private; attachable to your own agent configs) |
| `GET /agent/tools` + CRUD | Bearer (admin) | Manage MCP tool servers (applied on restart or `/agent/tools/reconnect`) |
| `POST /agent/tools/reconnect` | Bearer (admin) | Reconnect MCP servers from the store + rebuild the agent |
| `GET/POST /agent/connections` + `GET/PUT/DELETE /agent/connections/{name}` | read: any user; write: Bearer (admin) | Manage API connections (base URL + API key in the DB, used by the agent instead of `.env` keys; key is write-only, updates merge) |
| `POST /mcp/tools/call` | Bearer | MCP apps tools proxy: invoke a tool on a configured MCP server (`server_hint` or fan-out; `CallToolResult` passthrough — 200 with `isError`, 502 transport, 404 no match) |
| `GET|POST /agents` | Bearer | List / create agent configs (profile: model + system prompt + skills + tools; `scope: "global"` requires admin) |
| `GET/PUT/DELETE /agents/{name}` | Bearer | Read / replace / delete an agent config (owner; global ones: admin) |
| `POST /agents/{name}/test` | Bearer | Dry-run: build the graph (validates the model string); `400` when it cannot be built |
| `GET /users/me` | Bearer | Current user |
| `POST /users/me/password` | Bearer | Change your own password (old password must verify) |
| `GET /users` | Bearer (admin) | List all users (no password hashes) |
| `POST /users` | Bearer (admin) | Create a user (admin may grant the admin role) |
| `PATCH /users/{username}` | Bearer (admin) | Change a user's role and/or disabled state |
| `DELETE /users/{username}` | Bearer (admin) | Delete a user (their threads/history stay, orphaned) |
| `GET /health` | – | Status: persistence backend, MCP servers, model, interrupt_on, searxng, execute, agent_resources |

### Auth

On first start with an empty users store, a default admin account is seeded:
`admin` / `admin` (override via `DEFAULT_ADMIN_USERNAME` / `DEFAULT_ADMIN_PASSWORD`).
Users are stored in Postgres (`users` table, created by the SQL migrations in
`migrations/`), or in-memory when `DATABASE_URI` is unset (dev mode).

Roles: `user` (default — can chat, read own threads) and `admin` (manages
users via `GET /users` / `PATCH /users/{username}` and all agent resources:
skills, MCP tool servers). Registration always creates a `user`; promotion is
admin-only. Disabled accounts are rejected on every authenticated request.

Tokens: `POST /login` returns an access token (30 min) **plus a refresh
token** (7 days); `POST /refresh` exchanges a refresh token for a fresh
access token. Refresh tokens are stateless JWTs (no revocation store) —
logout is client-side discard of both tokens.

Login brute-force protection: failed logins are rate-limited per client IP +
username (in-memory sliding window, `LOGIN_RATE_LIMIT_MAX` /
`LOGIN_RATE_LIMIT_WINDOW`, default 10 per 15 min → `429`).

### Chat

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/login \
  -d "username=admin&password=admin" | jq -r .access_token)

curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "What is in my workspace?", "thread_id": null}'
```

- Omit `thread_id` (or pass `null`) to start a new conversation; the new id is
  returned in the final `done` event.
- Pass an existing `thread_id` to continue a conversation (Postgres checkpointing).
- Optional `enable_search` field overrides the SearXNG toggle per request
  (`null`/absent = use `SEARXNG_ENABLED` config).

### SSE events

```
event: message_delta   {"id","delta"}                        token chunk
event: message         {"id","message"}                      finalized message (langchain schema)
event: tool_start      {"id","name","args"}                  tool call began
event: tool_delta      {"id","name","delta"}                 tool output chunk
event: tool_end        {"id","name","output","is_error"}     tool finished (output = ToolMessage)
event: subagent        {"name","status","output"?,"error"?}  delegated task lifecycle
event: subagent_delta  {"subagent","delta"}                  subagent token chunk
event: interrupt       {"thread_id","interrupts"[]}          run paused for human approval
event: error           {"source","message"}                  recoverable error
event: done            {"thread_id","messages"[],"interrupted"?,"cancelled"?,"usage"?}  final state
```

The `done` event may carry `cancelled: true` (run aborted via
`POST /threads/{id}/cancel`) and `usage` (summed `input_tokens` /
`output_tokens` / `total_tokens` across the run's finalized messages, when
the model reports usage metadata).

Frontend plan: `message_delta` renders the streaming bubble; `tool_*` events can
render tool-call chips and, later, interactive components from MCP structured
content (`tool_end.output.content` may contain multiple blocks); `subagent_*`
events render delegation cards; `done.messages` is the source of truth to
persist client-side (note: checkpointed message `content` is a plain string,
while streamed `message` events use content blocks).

### AI SDK endpoint (`POST /api/chat`)

The Vercel AI SDK chat frontend (`ai-frontend-lab`) talks to this endpoint
instead of the raw SSE contract above. Request body:

```json
{
  "id": "chat-uuid",
  "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}],
  "selectedChatModel": "openai:deepseek-v4-flash",
  "agent": "research"
}
```

- The last user message runs through the agent; `id` is reused as the
  `thread_id` so conversations continue.
- Optional `enableSearch` field overrides the SearXNG web search toggle per
  chat (frontend search switch).
- Response is the AI SDK data-stream protocol (SSE `data:` chunks:
  `start`, `text-*`, `custom` for tool/subagent/interrupt activity, `finish`, `[DONE]`).
- Auth is optional for now: a Bearer JWT scopes thread metadata to that user;
  without one, a `guest` namespace is used (the frontend has no login yet).
- **Human-in-the-loop**: when a run pauses for approval, the stream emits a
  `custom` chunk `{"kind": "app.interrupt", "providerMetadata": {"app":
  {"threadId", "interrupts"}}}` (payload nested under the provider key, as
  the AI SDK requires) and ends with `finish` (`finishReason: "other"`).
  The frontend shows the approval UI, then resumes by posting to `/api/chat`
  again with the same `id` plus `decision` (or `decisions`):

  ```json
  {"id": "<thread id>", "decision": {"type": "approve"}, "messages": []}
  ```

  Decision types match `POST /threads/{id}/resume`: `approve`, `edit`,
  `reject`, `respond`. Resuming a thread that is not waiting returns `409`.
  Cancelling an in-flight run: `POST /threads/{id}/cancel` (Bearer; 409 when
  nothing is running).

## Agent configs (customizable agent profiles)

Beyond picking a model, you can define **named agent profiles** that bundle
model + system prompt + a selection of skills and MCP tool servers, then
reference them from chat requests via the `agent` field:

```json
// POST /agents  (Bearer)
{
  "name": "research",
  "model": "anthropic:claude-sonnet-4-5",
  "system_prompt": "You are a research assistant. Cite sources.",
  "skills": ["sql-guru"],
  "tools": ["weather", "web_search"],
  "temperature": 0.3
}
```

- **Resolution**: a chat request picks the caller's own agent with that
  name, then a global one, then the builtin `default` (env-driven,
  `DEEPAGENTS_MODEL` + `SYSTEM_PROMPT`). The name `default` is reserved.
- **Semantics**: `skills: null` / `tools: null` inherit the global behavior
  (all global skills / all configured MCP tools); `[]` disables them; a
  non-empty list selects. `web_search` is a built-in pseudo-tool name.
- **Skills**: selecting a skill snapshot-copies its `SKILL.md` + bundled
  files into the agent's own namespace (`/skills/<owner>/<name>/` backend
  route), so the agent sees exactly its selection. Skill names resolve
  against **your own skills first** (`/skills` CRUD), then global skills;
  global agents can only reference global skills. Editing the source skill
  later does not propagate to existing snapshots (re-save the agent to
  re-sync).
- **Per-thread**: thread metadata records the agent, so `resume` and thread
  history use the same agent. `GET /threads` returns the `agent` field.
- **Scope**: `scope: "user"` (default) agents are private; `scope:
  "global"` agents are shared by all users and require admin.
- **Implementation note**: the deep-agents graph bakes model + prompt +
  skills in at build time, so graphs are built lazily per distinct config
  and cached (`AgentRegistry`); skills/tools/config CRUD invalidates the
  cache. Conversations survive rebuilds (shared checkpointer).

## MCP servers (gofastmcp)

Server config comes from `MCP_SERVERS_JSON` (env) or `mcp_servers.json` (see
`mcp_servers.json.example`):

```json
{
  "weather": {
    "url": "http://localhost:8090/mcp",
    "transport": "streamable_http",
    "headers": {"Authorization": "Bearer xxx"}
  },
  "local-cli": {
    "command": "/path/to/gofastmcp-tool",
    "args": ["serve"],
    "transport": "stdio",
    "env": {"FOO": "bar"}
  }
}
```

- `streamable_http` — for gofastmcp servers deployed as web services
  (the recommended production transport; SSE transport also exists in gofastmcp).
- `stdio` — for gofastmcp binaries run as subprocesses.
- Tools are fetched at startup and merged into the agent alongside the built-in
  filesystem/subagent tools. Anything an MCP tool returns (text, structured
  content, multimodal blocks) is serialized into `tool_end` events.

Quick local MCP test (protocol-identical to a gofastmcp server):

```bash
uv run python scripts/test_mcp_server.py   # exposes get_weather/list_cities at :8090/mcp
MCP_SERVERS_JSON='{"weather-demo":{"url":"http://127.0.0.1:8090/mcp","transport":"streamable_http"}}' \
  uv run uvicorn app.main:app --port 8000
# GET /health -> mcp_servers: ["weather-demo"]; ask the agent about the weather in Jakarta
```

## Web search (SearXNG)

The agent gets a `web_search` tool backed by a self-hosted
[SearXNG](https://docs.searxng.org/) metasearch instance (no API key, no
per-query cost — SearXNG aggregates Google/Bing/DDG/... for you). JSON output
is enabled via the mounted `searxng/settings.yml` (`search.formats: [html, json]`).

```bash
docker compose up -d searxng        # http://localhost:8080
# .env:
SEARXNG_URL=http://localhost:8080
SEARXNG_ENABLED=true
```

Toggle levels:

| Level | Switch | Effect |
|---|---|---|
| Install | `SEARXNG_URL` set | `web_search` tool is registered on the agent (unset = invisible, zero overhead) |
| Global | `SEARXNG_ENABLED=true/false` | Tool exists but returns "Web search is disabled for this request." |
| Per request | `"enable_search": false` in `POST /chat` / `"enableSearch": false` in `POST /api/chat` | Frontend toggle per message; overrides the config |

`GET /health` reports `"searxng": {"installed": bool, "enabled": bool}`.

## Shell execution (execute tool)

The agent's built-in `execute` tool runs shell commands, but only when the
backend supports execution — so it's **opt-in and off by default**:

```bash
# .env — dev/trusted environments ONLY (unrestricted, irreversible commands)
EXECUTE_ENABLED=true
EXECUTE_MAX_TIMEOUT=3600        # cap on per-command timeout (seconds)
EXECUTE_INHERIT_ENV=false       # true = shell sees server env (incl. secrets)
```

When enabled, the default `StateBackend` is swapped for deepagents'
`LocalShellBackend` (commands run directly on the host with your user's
permissions). Safety guidance from upstream: pair it with human approval
(`INTERRUPT_ON_JSON='{"execute": true}'`), run only in dedicated dev
environments, and never on a shared/production server — tool-level command
permissions are not implemented upstream. `GET /health` reports
`"execute": {"enabled": bool, "max_timeout": int}`.

## Postgres

Start Postgres, then set `DATABASE_URI`:

```bash
docker compose up -d postgres        # or: brew install postgresql@16 && brew services start postgresql@16
# export DATABASE_URI=postgresql://aibackend:aibackend@localhost:5432/aibackend
```

Tables are created automatically at startup (`setup()`). Without `DATABASE_URI`
the app runs on in-memory checkpointer + store (data lost on restart).

- **Threads** live in the checkpointer; thread metadata (title, timestamps) in the
  store under namespace `("threads", <username>)` → per-user listing.
- **Chat history**: every run also writes readable rows to a dedicated
  `chat_messages` table (thread_id, username, role, message content as JSONB,
  created_at; deduped by message id so resume/retries never duplicate).
  `GET /threads/{id}/messages` serves from this table, falling back to
  checkpoint rehydration for older threads. Query it directly, e.g.
  `SELECT thread_id, role, created_at FROM chat_messages ORDER BY id DESC LIMIT 20;`
- **Durable workspace**: the agent's filesystem backend is a `StoreBackend` over
  the LangGraph store — every user's files (scratch space, memories) persist
  across threads and restarts, scoped per user (`("<username>")` namespace).
  With `EXECUTE_ENABLED=true` the default backend becomes `LocalShellBackend`
  (host filesystem + shell) instead.
- **Skills**: agent-wide skills live under the global `("agent", "skills")`
  namespace as `/<name>/SKILL.md` plus optional bundled files (scripts/,
  references/, assets/...), shared by all users. The agent loads them from the
  `/skills/` backend source on every run — edits via the API apply without a
  restart.

## Agent resources API

Skills and MCP tool servers are managed via CRUD endpoints and persisted in the
store (Postgres in production). Namespaces: `("agent", "skills")` for skills,
`("agent", "mcp_servers")` for tool servers.

Skills follow the deepagents **skill-creator layout**: a required `SKILL.md`
plus optional bundled resources (`scripts/`, `references/`, `assets/`, ... —
any relative path, so `eval/` works too). Bundled files land under
`/<name>/<path>` in the store, so the agent's filesystem tools (`ls`, `read`,
`write`, `execute`) see the whole skill folder. `PUT` replaces the listed
files and keeps unlisted ones; `DELETE /agent/skills/{name}/files/{path}`
removes a single file.

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/login -d "username=admin&password=admin" | jq -r .access_token)
AUTH="Authorization: Bearer $TOKEN"

# --- skills (skill-creator layout: SKILL.md + optional scripts/, references/, assets/) ---
curl -X POST http://127.0.0.1:8000/agent/skills -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"name":"release-notes","description":"Write concise release notes","content":"## Steps\\n1. git log --oneline\\n2. group by type",
       "files":[{"path":"scripts/release.py","content":"print(\"hi\")"},{"path":"references/style.md","content":"# Style guide"}]}'
curl http://127.0.0.1:8000/agent/skills -H "$AUTH"                       # list
curl http://127.0.0.1:8000/agent/skills/release-notes -H "$AUTH"          # get (includes files)
curl -X PUT  http://127.0.0.1:8000/agent/skills/release-notes -H "$AUTH" -H 'Content-Type: application/json' -d '{...}'
curl -X DELETE http://127.0.0.1:8000/agent/skills/release-notes -H "$AUTH"
curl -X DELETE http://127.0.0.1:8000/agent/skills/release-notes/files/scripts/release.py -H "$AUTH"

# --- MCP tool servers (same shape as mcp_servers.json) ---
curl -X POST http://127.0.0.1:8000/agent/tools -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"name":"weather","transport":"streamable_http","url":"http://localhost:8090/mcp"}'
curl -X POST http://127.0.0.1:8000/agent/tools/reconnect -H "$AUTH"      # apply tool changes live
```

- Skill changes are picked up by the agent on the **next run** (no rebuild).
- Tool server changes apply on **restart** or `POST /agent/tools/reconnect`.
- When the store has tool servers, they **replace** the `MCP_SERVERS_JSON` /
  `mcp_servers.json` env config; delete all entries to fall back.

## Knowledge bases (RAG)

Per-user knowledge bases: upload files (or folders — send each file with its
relative `path`), the backend extracts text, chunks it and embeds it into
**Weaviate** (hybrid BM25F + vector search). The agent gets a
`search_knowledge_base` tool so it can answer from uploaded documents during
chat; the tool only ever sees the current user's KBs (the run context carries
`user_id`, which also activates per-user workspace isolation).

```bash
# Start the vector store once
# docker compose up -d weaviate

export AUTH="Authorization: Bearer $(curl -s -X POST http://127.0.0.1:8000/login -d 'username=admin&password=admin' | jq -r .access_token)"

# create a KB + upload a folder (file + relative path pairs)
KB=$(curl -s -X POST http://127.0.0.1:8000/kb -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"name":"runbook","description":"Ops docs"}' | jq -r .id)
curl -X POST "http://127.0.0.1:8000/kb/$KB/files" -H "$AUTH" \
  -F "file=@guides/deploy.md;type=text/markdown" -F "paths=guides/deploy.md" \
  -F "file=@guides/backup.md;type=text/markdown" -F "paths=guides/backup.md"

# ...or one zip for a whole folder tree
curl -X POST "http://127.0.0.1:8000/kb/$KB/zip" -H "$AUTH" -F "file=@docs.zip"

curl "http://127.0.0.1:8000/kb/$KB/files" -H "$AUTH"                       # status per file
curl "http://127.0.0.1:8000/kb/$KB/search?q=kubectl%20deployment" -H "$AUTH"  # hybrid search
curl "http://127.0.0.1:8000/kb/search?q=deployment" -H "$AUTH"             # search all my KBs
curl -o deploy.md "http://127.0.0.1:8000/kb/$KB/files/<doc_id>/content" -H "$AUTH"  # raw file
curl -X POST "http://127.0.0.1:8000/kb/$KB/reindex" -H "$AUTH"             # re-embed everything
curl -X DELETE "http://127.0.0.1:8000/kb/$KB" -H "$AUTH"                   # delete KB + vectors
```

- Supported extensions: `.md .txt .pdf .docx .csv .html .json` + common code
  files (configurable via `KB_ALLOWED_EXTENSIONS`); cap 25 MB/file
  (`KB_MAX_FILE_SIZE_MB`). PDFs are chunked **page-level** (NVIDIA benchmark:
  best average retrieval accuracy); markdown is split on headers.
- Zip uploads are guarded: path traversal, entry count (`KB_ZIP_MAX_ENTRIES`)
  and total uncompressed size (`KB_ZIP_MAX_TOTAL_MB`) are rejected before
  anything is stored; per-entry extension/quota issues produce per-entry
  results.
- Per-user storage quota: `KB_QUOTA_MB` (default 500 MB, sum of raw bytes).
- Hybrid search tuning: `KB_HYBRID_ALPHA` (0 = keyword, 1 = vectors, default
  0.5) and per-request `?alpha=` on both search endpoints;
  `KB_BM25_PROPERTY_WEIGHTS` (default `{"path": 2.0}`) boosts titles/paths in
  the BM25F stage. Embedding dimensions via `EMBEDDINGS_DIMENSIONS`
  (Matryoshka truncation); switching embedding models requires a
  `POST /kb/{id}/reindex`.
- Reranking (retrieve broad → rerank fine): set `KB_RERANK_MODEL` to a
  flashrank model name (e.g. `ms-marco-MiniLM-L-12-v2`, a tiny CPU
  cross-encoder, ~30ms for 20 candidates; model downloads on first use).
  Retrieval pulls `KB_RERANK_CANDIDATES` (default 20) then reranks down to
  the requested limit. Unset → plain hybrid search.
- Query rewriting (opt-in, `KB_QUERY_REWRITE=true`): an LLM call rewrites
  vague queries before retrieval (`KB_REWRITE_MODEL`, defaults to the agent
  model; queries shorter than `KB_REWRITE_MIN_LENGTH` or single tokens are
  left untouched; results cached per query; failures degrade to the original
  query).

### Live reranker check

```bash
uv run python scripts/test_reranker.py   # real FlashRank vs store ranking on a demo corpus
```

Downloads the model once (~22 MB to /tmp) and prints per-query before/after
rankings so you can eyeball whether reranking changes orders sensibly.

### Retrieval evaluation (golden set)

Before tuning anything, build a golden set of real queries + relevant
document paths and measure. See `data/golden_set.example.json` for the
format and `docs/rag-techniques-research.md` for why this comes first.

```bash
# in-memory sweep (uses configured embeddings; no Weaviate needed)
uv run python scripts/kb_eval.py --kb runbook --owner admin --golden data/golden_set.json

# against the live Weaviate store
uv run python scripts/kb_eval.py --kb runbook --golden data/golden_set.json --live

# per-query hit lists
uv run python scripts/kb_eval.py --kb runbook --golden data/golden_set.json --verbose
```

Output: Recall@k, MRR and nDCG@k per alpha value, plus the best alpha.
Every future retrieval change should be gated on these numbers.

```bash
# compare reranking vs plain retrieval on the same golden set
uv run python scripts/kb_eval.py --kb runbook --golden data/golden_set.json --rerank
```
- Ingest status per document: `pending → processing → ready | failed` (with
  error message).
- Embeddings: OpenAI (`EMBEDDINGS_MODEL`, default `text-embedding-3-small`)
  when `OPENAI_API_KEY` is set; otherwise a deterministic local embedder for
  dev/tests.
- Without `WEAVIATE_URL`, upload/search return 503 and the agent has no KB
  tool — the rest of the app is unaffected.

## Code layout

Follows the [fastapi-clean-architecture](https://github.com/jujumilk3/fastapi-clean-architecture)
template: versioned endpoints under `api/`, core infrastructure, per-domain
schema modules, and a service layer.

```
src/app/          package (src layout, installed editable by uv sync)
  main.py         thin FastAPI factory: lifespan (agent startup), CORS, routers
  api/v1/
    routes.py         router aggregation for API v1
    endpoints/        one router per domain
      auth.py             /login, /users/me
      chat.py             /chat (SSE), /api/chat (AI SDK), /threads
      agent.py            /agent/skills + /agent/tools CRUD, reconnect
      health.py           /health
  core/           cross-cutting infrastructure
    config.py         settings (env-driven, sectioned)
    constants.py      store namespaces, SSE headers, default system prompt
    database.py       Postgres checkpointer + store + chat_messages + users (in-memory fallback)
    migrations.py     SQL migration runner (applies migrations/*.sql at startup)
    dependencies.py   get_current_user (validates JWT against the users store)
    security.py       bcrypt + JWT (access + refresh tokens)
    rate_limit.py     in-memory sliding-window login limiter
    run_registry.py   active agent runs keyed by thread_id (cancel support)
    exceptions.py     HTTP exception hierarchy (NotFound, Conflict, ...)
  migrations/      SQL migrations (0001_create_users.sql, 0002_create_chat_messages.sql,
                   0003_create_user_roles.sql)
  schema/         per-domain API models
    auth_schema.py    Token, TokenData, User, UserInDB, UserRole, UserUpdate
    chat_schema.py    ChatRequest, ThreadOut, AiSdkChatRequest, ResumeRequest
    agent_schema.py   SkillIn/Out, ToolServerIn/Out
  services/       business logic
    agent.py      create_deep_agent factory + shared durable backend
    chat.py       agent streaming -> normalized SSE events (normative contract)
    ai_sdk_chat.py  AI SDK data-stream protocol bridge
    searxng.py    web_search tool (toggleable client + tool factory)
    mcp.py        MultiServerMCPClient (store-first, streamable_http / stdio)
    resources.py  skills + MCP tool server CRUD on the durable store
  util/
    date.py       time helpers (now_iso)
tests/            offline pytest suite (scripted model, no API key): uv run pytest -q
scripts/
  live_test.py       live E2E against a running server + real model
  test_mcp_server.py tiny MCP server (streamable HTTP) to test MCP wiring
```

Conventions live in `AGENTS.md` (ruff, git workflow, structure).

## Human-in-the-loop

Set `INTERRUPT_ON_JSON` to pause runs before sensitive tool calls:

```bash
INTERRUPT_ON_JSON='{"write_file": true, "edit_file": true}' uv run uvicorn app.main:app --port 8000
```

When a paused tool is requested, the stream emits:

```
event: interrupt
data: {"thread_id": "...", "interrupts": [{"action_requests": [{"name": "write_file", "args": {...}, "description": "..."}], "review_configs": [...]}]}

event: done
data: {"thread_id": "...", "messages": [...], "interrupted": true}
```

The frontend shows an approval UI, then calls `POST /threads/{id}/resume`:

```bash
curl -N -X POST http://127.0.0.1:8000/threads/<id>/resume \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"decision": {"type": "approve"}}'
```

Decision types (one per `action_request`; use `decisions: [...]` for several):

| type | body |
|---|---|
| `approve` | run the tool as requested |
| `edit` | `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}` |
| `reject` | `{"type": "reject", "message": "..."}` — tool skipped, model informed |
| `respond` | `{"type": "respond", "message": "..."}` — answer on behalf of the tool |

Resuming a thread that is not waiting returns `409`. The resumed run streams
normal events (`tool_start`/`tool_end`, `message_delta`, ...) until `done`.

### Sharing chats

Threads can be shared as **public, read-only links** (no auth needed to
view):

```bash
curl -X POST http://127.0.0.1:8000/threads/<id>/share -H "Authorization: Bearer $TOKEN"
# {"share_token": "<unguessable-token>", "url": "http://127.0.0.1:8000/shared/<token>"}

curl http://127.0.0.1:8000/shared/<token>   # no auth: thread_id, title, username, messages
```

- Share tokens are random 32-byte URL-safe strings; sharing is idempotent
  (re-POST returns the existing token). Revoking (`DELETE
  /threads/{id}/share`) or deleting the thread kills the link immediately.
- `GET /threads` includes `share_token` so the frontend can render the
  share state per thread; only the owner can create/read/revoke a link.
