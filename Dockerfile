# syntax=docker/dockerfile:1

# FastAPI backend: Deep Agents harness + Postgres persistence + MCP tools.
# Official uv image (Python 3.12, Debian bookworm slim): locked dependency set
# without the dev group, installed in two cached layers; uvicorn runs as a
# non-root user with a healthcheck.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# The execute tool inherits this PATH (EXECUTE_INHERIT_ENV=true), so the
# agent's shell resolves `python`/`uv`/`uvicorn` like a dev shell.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Locked dependencies only — layer cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Application code (src layout, installed editable by uv) + SQL migrations
#    (applied at startup by core/migrations.py, resolved relative to /app).
COPY src ./src
COPY migrations ./migrations
RUN uv sync --frozen --no-dev

# 3) Non-root runtime user; uploads + agent workspace files land under
#    /app/uploads (mounted as a named volume in compose so they survive
#    app container recreates).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/uploads \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
