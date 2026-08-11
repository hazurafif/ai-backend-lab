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

## Code style

- **Ruff is the only linter/formatter.** Config in `pyproject.toml` — line-length
  100, target py3.12. No blanket `noqa`; fix the code instead.
- Type hints on all public functions; `from __future__ import annotations` for
  forward references. Prefer dataclasses/pydantic models over ad-hoc dicts.
- **Async-first** — never block the event loop in request paths.
- Concise Google-style docstrings; the SSE event contract in `app/main.py`'s
  module docstring is normative — keep it in sync.

## Structure

- **src layout**: package in `src/app/` (installed editable via `uv sync`) —
  never import from the repo root.
- `main.py` thin app factory; routes in `api/` (`routes_auth`, `routes_chat`,
  `routes_agent`, `routes_health`); business logic in `services/` (`agent`,
  `chat`, `mcp`, `searxng`, `resources`); settings/constants in `core/`;
  `db.py` persistence, `schemas.py` API models, `exceptions.py` HTTP errors.
- New routes: add an `APIRouter` in `api/` (or a new `routes_*.py` module)
  and include it in `api/__init__.py`.
- Offline tests in `tests/`; live helpers in `scripts/`.

## Testing

- `uv run pytest -q` must be green before every commit.
- Offline tests: no network, no API keys, no Postgres.
- Changes to the agent pipeline (streaming, middleware, persistence) → extend
  `tests/test_smoke.py` with a scripted-model test.

## Git workflow

- **Conventional commits** (enforced by pre-commit):

```
<type>(<scope>): imperative summary      # ≤72 chars, lowercase, no period

<why + what, not how>
```

  Types: `feat fix chore docs style refactor perf test build ci revert`.
- Before every commit: `ruff check . && ruff format .` → `pytest -q` → `git add`
  only the files belonging to the change.
- Push your own branch (`git push -u origin <branch>`), open a PR. Never
  force-push `main`; prefer `git pull --rebase` over merges.

## Security

- Never commit `.env`, secrets, or generated files (see `.gitignore`).
- "Trust the LLM" model: enforce boundaries via permissions/sandboxes, not
  prompt instructions.
