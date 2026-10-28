-- Runs once, when the Postgres data volume is first created.
-- Enables pgvector for embeddings. (PGVector also tries this at startup,
-- but doing it here means the app user needs no superuser rights.)
CREATE EXTENSION IF NOT EXISTS vector;
