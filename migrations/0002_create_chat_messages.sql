-- Chat history rows: one row per message, readable via SQL.
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    username    TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (thread_id, message_id)
);
CREATE INDEX IF NOT EXISTS ix_chat_messages_thread
    ON chat_messages (thread_id, created_at);
