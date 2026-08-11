-- Roles: "admin" (full access, incl. agent resources) vs "user" (default).
-- Existing installs keep their users; the seeded default admin is promoted
-- to admin at startup (see Persistence.ensure_default_admin).
ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';
