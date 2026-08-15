-- Per-user preferences (key-value JSON), one row per user.
-- Keys: 'enable_search' -> bool (web search toggle; unset = SEARXNG_ENABLED
-- server default). Owned by the users table; deleting a user drops the row.
CREATE TABLE IF NOT EXISTS user_preferences (
    username   TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
