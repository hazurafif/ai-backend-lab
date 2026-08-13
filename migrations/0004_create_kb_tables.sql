-- Knowledge bases (per-user) + ingested documents (raw blobs).
-- Vectors live in Weaviate (KnowledgeChunk collection); this table holds
-- metadata + the raw file bytes so backup stays single-point (Postgres).
CREATE TABLE IF NOT EXISTS kb (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner       TEXT NOT NULL,              -- username; owner-only access
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner, name)
);

CREATE TABLE IF NOT EXISTS kb_documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id       UUID NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    owner       TEXT NOT NULL,
    path        TEXT NOT NULL,              -- relative path, supports folders
    mime_type   TEXT,
    size_bytes  BIGINT NOT NULL,
    content     BYTEA NOT NULL,             -- raw file bytes
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|ready|failed
    error       TEXT,
    chunk_count INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kb_id, path)
);

CREATE INDEX IF NOT EXISTS ix_kb_documents_kb ON kb_documents (kb_id);
CREATE INDEX IF NOT EXISTS ix_kb_documents_owner ON kb_documents (owner);
