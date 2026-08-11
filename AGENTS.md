# AGENTS.md — project conventions

AI backend service: FastAPI wrapping **LangChain Deep Agents**, Postgres
persistence, MCP tool integration, and SSE streaming.

## Commands

```bash
uv sync                                     # install deps (Python 3.12)
uv run uvicorn app.main:app --port 8000 --reload
uv run ruff check . && uv run ruff format . # lint + format (pre-commit enforced)
uv run pytest -q                            # offline suite, no API key needed
uv run python scripts/live_test.py          # live E2E (needs running server + key)
```

## Architecture

Follows the
[fastapi-clean-architecture](https://github.com/jujumilk3/fastapi-clean-architecture)
template: versioned endpoints in `api/`, cross-cutting infrastructure in
`core/`, per-domain schemas in `schema/`, business logic in `services/`,
helpers in `util/`.

```
src/app/          package (src layout, installed editable by uv sync)
  main.py         thin FastAPI factory: lifespan (agent startup), CORS, routers
  api/v1/
    routes.py         aggregates all endpoint routers into api_router
    endpoints/        one router per domain
      auth.py         /login, /users/me
      chat.py         /chat (SSE), /api/chat (AI SDK), /threads
      agent.py        /agent/skills + /agent/tools CRUD, reconnect
      health.py       /health
  core/           cross-cutting infrastructure
    config.py         settings (env-driven, sectioned)
    constants.py      store namespaces, SSE headers, default system prompt
    database.py       Postgres checkpointer + store + chat_messages history + users (in-memory fallback)
    migrations.py     SQL migration runner (applies migrations/*.sql at startup)
    dependencies.py   get_current_user (loads the user from the users store)
    security.py       bcrypt + JWT
    exceptions.py     HTTP exception hierarchy (NotFound, Conflict, ...)
  migrations/      SQL migrations (0001_create_users.sql, 0002_create_chat_messages.sql)
  schema/         per-domain API models
    auth_schema.py    Token, TokenData, User, UserInDB
    chat_schema.py    ChatRequest, ThreadOut, AiSdkChatRequest, ResumeRequest
    agent_schema.py   SkillIn/Out, ToolServerIn/Out
  services/       business logic
    agent.py          create_deep_agent factory + shared durable backend
    chat.py           agent streaming -> normalized SSE events (normative contract)
    ai_sdk_chat.py    AI SDK data-stream protocol bridge
    mcp.py            MultiServerMCPClient (store-first, streamable_http / stdio)
    resources.py      skills (SKILL.md + bundled files) + MCP tool server CRUD on the durable store
    searxng.py        web_search tool (toggleable client + tool factory)
  util/
    date.py           time helpers (now_iso)
tests/            offline pytest suite (scripted model, no API key)
scripts/          live E2E + MCP test server helpers
```

Rules:

- **New endpoint** → add a router module in `api/v1/endpoints/` and include it
  in `api/v1/routes.py`. Don't invent new router folders.
- **New schema** → `schema/<domain>_schema.py` (Pydantic v2, `Field(...)`
  constraints). One module per domain, not one big `schemas.py`.
- **New business logic** → `services/`; keep routers thin (validate + call
  service + respond). No repository layer: persistence is the LangGraph
  checkpointer/store singleton in `core/database.py`.
- The SSE event contract in `app/main.py`'s module docstring is **normative** —
  keep it in sync with `services/chat.py`.
- Cross-module imports use explicit module paths (e.g.
  `from app.services import resources`), never `import *`.
- Never import from the repo root — only from `src/app/` (editable install).

## Code style

- **Ruff is the only linter/formatter.** Config in `pyproject.toml` — line-length
  100, target py3.12. No blanket `noqa`; fix the code instead.
- Type hints on all public functions; `from __future__ import annotations` for
  forward references. Prefer pydantic models over ad-hoc dicts.
- Concise Google-style docstrings.

## Async rules

- `async def` for awaitable I/O; plain `def` for blocking I/O (FastAPI runs it
  in a threadpool); `await run_in_threadpool(...)` for a sync call inside an
  async route.
- Never call blocking code (`time.sleep`, `requests.get`, sync DB drivers,
  `open()`) inside `async def` — it freezes the event loop.
- Dependencies and routes: prefer `async def` unless the body is blocking.

## Dependencies & schemas

- Use `Annotated[T, Depends(...)]` for new route dependencies (modern FastAPI
  form). Existing legacy `= Depends(...)` defaults are tolerated.
- `Depends` is cached per request — reuse dependencies instead of re-querying.
- Validate inside dependencies (load + validate + return), not in the route
  body.
- Pydantic v2 only: no `json_encoders` (use `@field_serializer`), no `.dict()`.

## Testing

- `uv run pytest -q` must be green before every commit.
- Offline tests: no network, no API keys, no Postgres (in-memory checkpointer
  + store). Use `httpx.AsyncClient` + `ASGITransport`, never
  `async_asgi_testclient`.
- Override auth in tests with `app.dependency_overrides`, not monkeypatching.
- Changes to the agent pipeline (streaming, middleware, persistence) → extend
  `tests/test_smoke.py` with a scripted-model test.

## Git workflow

- **Conventional commits** (enforced by pre-commit):

```
<type>(<scope>): imperative summary      # ≤72 chars, lowercase, no period

<why + what, not how>
```

  Types: `feat fix chore docs style refactor perf test build ci revert`.
- **Auto-commit**: after implementing a requested change and seeing the suite
  green, commit it yourself (conventional format above) instead of leaving
  changes uncommitted for the user. Split into logical commits when the change
  spans distinct concerns; never commit unrelated files.
- Before every commit: `ruff check . && ruff format .` → `pytest -q` → `git add`
  only the files belonging to the change.
- Push your own branch (`git push -u origin <branch>`), open a PR. Never
  force-push `main`; prefer `git pull --rebase` over merges.

## Security

- Never commit `.env`, secrets, or generated files (see `.gitignore`).
- "Trust the LLM" model: enforce boundaries via permissions/sandboxes, not
  prompt instructions.
