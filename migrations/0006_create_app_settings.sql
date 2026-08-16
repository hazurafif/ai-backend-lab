-- App settings: key-value admin configuration overrides (DB wins over .env).
-- Keys: 'execute' -> {"enabled": bool, "max_timeout": int, "inherit_env": bool}
--       'hitl'    -> {"interrupt_on": {tool: bool}}
-- Credentials have NO env fallback: the agent LLM and KB embeddings resolve
-- DB connections only (see 0005) and fail loudly when none exists.
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
