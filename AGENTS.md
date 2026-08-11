# AGENTS.md

Project conventions for AI agents and humans working in this repository.

## Project overview

AI backend service: a FastAPI API wrapping **LangChain Deep Agents**
(`deepagents`) as the core agent, with Postgres persistence
(`langgraph-checkpoint-postgres`), MCP tool integration (gofastmcp servers via
`streamable_http`/`stdio`), and SSE streaming for a future frontend.

## Commands

```bash
uv sync                        # install dependencies (Python 3.12, pinned in .python-version)
uv run uvicorn app.main:app --port 8000 --reload   # run the API
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pre-commit run --all-files   # run all git hooks
uv run pytest -q               # offline test suite (no API key needed)
uv run python scripts/live_test.py           # live E2E (needs running server + model key)
uv run python scripts/test_mcp_server.py     # tiny MCP server for MCP integration testing
```

## Code style (ruff)

- **Ruff is the only linter and formatter.** Run `ruff check .` and
  `ruff format .` before every commit. Pre-commit hooks enforce both.
- Configuration lives in `pyproject.toml` under `[tool.ruff]` —
  line-length 100, target Python 3.12. Do not add per-file `noqa` suppressions
  without justification; prefer fixing the code.
- Type hints on all public functions and methods. Use `from __future__ import
  annotations` in modules with forward references.
- Prefer dataclasses / pydantic models over ad-hoc dicts for API shapes.
- Async-first: use `async def` and awaitable APIs (`langgraph` is async). Never
  block the event loop with sync I/O in request paths.
- Docstrings: concise Google-style for public modules, classes, and functions.
  The SSE event contract in `app/main.py`'s module docstring is normative —
  keep it in sync with the code.

## Project structure

- **src layout**: the package lives in `src/app/`. Never import from the
  repository root; the package is installed editable via `uv sync`.
- `src/app/main.py` — FastAPI app factory + SSE streaming (keep the event
  contract documented there).
- `src/app/agent.py` — agent factory (`create_deep_agent` wiring).
- `src/app/db.py` — persistence (Postgres checkpointer + store, in-memory fallback).
- `src/app/mcp_client.py` — MCP server connections.
- `src/app/config.py` — settings from `.env` (never hardcode secrets).
- `tests/` — offline tests (scripted model, no API key). Live/integration
  helpers live in `scripts/`.
- New routes: add to `main.py` until the router file exceeds ~10 endpoints,
  then split into `src/app/api/` with `APIRouter` modules.

## Testing

- Run `uv run pytest -q` before committing — it must be green.
- Offline tests must not require network, API keys, or a running Postgres.
- When changing the agent pipeline (streaming, middleware, persistence),
  extend `tests/test_smoke.py` with a scripted-model test.

## Git workflow

**Commit (conventional commits, enforced by pre-commit `commit-msg` hook):**

```
<type>(<scope>): <imperative summary>

<why + what, not how>
```

- Types: `feat`, `fix`, `chore`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `revert`. Scope is optional (e.g. `feat(chat): ...`).
- Subject: imperative mood, lowercase, ≤ 72 chars, no trailing period.
- Body: explain WHY, not just WHAT.
- Never commit `.env`, secrets, or generated files (see `.gitignore`).

**Before every commit:**

1. `uv run ruff check . && uv run ruff format .`
2. `uv run pytest -q`
3. `git add` only the files belonging to the change.

**Push:**

- Push your own feature branch with `git push -u origin <branch>`, then open a
  PR (or merge locally only when explicitly asked).
- Never force-push to `main`. Prefer rebase (`git pull --rebase`) over merge
  commits when syncing.
- If CI/lint fails after push, fix and re-push immediately.

## Security

- `.env` contains real credentials — never commit it, never paste values into
  chat/logs, and never echo secrets in code or docs.
- The agent follows a "trust the LLM" model; enforce boundaries via
  permissions/sandboxes, not prompt instructions.
