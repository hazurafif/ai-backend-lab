-- App settings: key-value admin configuration overrides (DB wins over .env).
-- Keys: 'execute' -> {"enabled": bool, "max_timeout": int, "inherit_env": bool}
--       'connections' -> {"fallback_env": bool}  (false = DB connections are
--       mandatory; .env credentials are never used)
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
