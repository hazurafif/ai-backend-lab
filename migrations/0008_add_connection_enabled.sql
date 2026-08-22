-- Connection enable/disable toggle (settings UI). Disabled connections are
-- excluded from default resolution, model discovery and agent binding — the
-- row stays visible for editing / re-enabling.
ALTER TABLE connections ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT true;

CREATE INDEX IF NOT EXISTS ix_connections_enabled_kind ON connections (enabled, kind);