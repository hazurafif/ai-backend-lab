# AI Backend — Deep Agents + Postgres + MCP (gofastmcp)

A FastAPI backend wrapping **LangChain Deep Agents** as the core agent:

- **Agent**: [`deepagents`](https://github.com/langchain-ai/deepagents) — planning,
  subagents (`task` tool), virtual filesystem, context management, skills & memory.
- **Persistence**: Postgres via `langgraph-checkpoint-postgres` —
  `AsyncPostgresSaver` (conversations) + `AsyncPostgresStore` (long-term memory,
  thread metadata). Falls back to in-memory when `DATABASE_URI` is unset.
- **MCP**: connects to MCP servers (e.g. built with **gofastmcp**) over
  `streamable_http` or `stdio` via `langchain-mcp-adapters`. MCP tool outputs are
  streamed to the frontend as structured events — ready to render as interactive
  UI elements (cards, charts, forms) later.
- **Web search**: a `web_search` tool backed by a self-hosted **SearXNG**
  metasearch instance, toggleable from config and per-request (frontend).
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

## Model configuration

`DEEPAGENTS_MODEL` accepts any `provider:model` string understood by
[`init_chat_model`](https://docs.langchain.com/oss/python/langchain/models)
(`openai:...`, `anthropic:...`, `google_genai:...`, ...). For a custom
OpenAI-compatible gateway set `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `.env`
(langchain-openai reads them) — e.g. the opencode.ai zen gateway.

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /login` | – | OAuth2 form `username`/`password` → JWT |
| `POST /chat` | Bearer | Run the agent; **SSE stream** of events |
| `POST /api/chat` | optional Bearer | AI SDK data-stream protocol for the frontend (`useChat`) |
| `GET /threads` | Bearer | Conversations of the current user (newest first) |
| `GET /threads/{id}/messages` | Bearer | Full history of a thread |
| `POST /threads/{id}/resume` | Bearer | Resume a run paused for human approval |
| `GET /users/me` | Bearer | Current user |
| `GET /health` | – | Status: persistence backend, MCP servers, model, interrupt_on, searxng |

### Chat

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/login \
  -d "username=johndoe&password=secret" | jq -r .access_token)

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
event: done            {"thread_id","messages"[],"interrupted"?}  final state
```

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
  "selectedChatModel": "openai:deepseek-v4-flash"
}
```

- The last user message runs through the agent; `id` is reused as the
  `thread_id` so conversations continue.
- Optional `enableSearch` field overrides the SearXNG web search toggle per
  chat (frontend search switch).
- Response is the AI SDK data-stream protocol (SSE `data:` chunks:
  `start`, `text-*`, `custom` for tool/subagent activity, `finish`, `[DONE]`).
- Auth is optional for now: a Bearer JWT scopes thread metadata to that user;
  without one, a `guest` namespace is used (the frontend has no login yet).
- Human-in-the-loop pauses surface as an `error` chunk (resume is not wired
  to this protocol yet).

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

## Postgres

Start Postgres, then set `DATABASE_URI`:

```bash
docker compose up -d postgres        # or: brew install postgresql@16 && brew services start postgresql@16
# export DATABASE_URI=postgresql+psycopg://aibackend:aibackend@localhost:5432/aibackend
```

Tables are created automatically at startup (`setup()`). Without `DATABASE_URI`
the app runs on in-memory checkpointer + store (data lost on restart).

- **Threads** live in the checkpointer; thread metadata (title, timestamps) in the
  store under namespace `("threads", <username>)` → per-user listing.
- **Memory**: the agent's `/memories/` filesystem path is routed to the store
  (`StoreBackend`), scoped per user, so agent memory survives across threads.

## Code layout

```
src/app/          package (src layout, installed editable by uv sync)
  main.py         FastAPI app + SSE streaming (event normalization)
  agent.py        create_deep_agent factory (persistence, MCP tools, memory backend)
  db.py           Postgres checkpointer + store (in-memory fallback)
  mcp_client.py   MultiServerMCPClient from config (streamable_http / stdio)
  searxng.py      SearXNG web_search tool (toggleable client + tool factory)
  config.py       settings from .env
  auth.py         bcrypt + JWT
  schemas.py      API models
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
