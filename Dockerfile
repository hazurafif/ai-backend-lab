FROM python:3.12-slim

# FastAPI backend: Deep Agents harness + Postgres persistence + MCP tools.
# Builds the locked dependency set with uv (no dev group) and runs uvicorn.

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
# SQL migrations are applied at startup (core/migrations.py resolves them
# relative to the repo root).
COPY migrations ./migrations

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
