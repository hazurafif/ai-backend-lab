-- Provider connections (base URL + API token) replacing .env credentials.
-- The agent LLM and KB embeddings resolve their "default" connection of the
-- matching kind (is_default=true, else first created) at startup and on CRUD.
CREATE TABLE IF NOT EXISTS connections (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'llm',   -- llm|embeddings|mcp|weaviate|searxng
    base_url    TEXT,
    api_token   TEXT,
    extra       JSONB NOT NULL DEFAULT '{}'::jsonb,  -- provider options (model, headers, ...)
    is_default  BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_connections_kind ON connections (kind);
