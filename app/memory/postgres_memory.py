"""PostgreSQL-backed chat history (``langchain_postgres.PostgresChatMessageHistory``).

Schema (created automatically by :func:`init_postgres_memory`)::

    CREATE TABLE chat_history (
        id         SERIAL PRIMARY KEY,
        session_id UUID NOT NULL,
        message    JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

``PostgresChatMessageHistory`` requires the session id to be a UUID, so
arbitrary client ids (e.g. ``"alice-1"``) are mapped to a stable UUIDv5.
"""

from __future__ import annotations

import threading
import uuid

import psycopg
from langchain_postgres import PostgresChatMessageHistory

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Fixed namespace so the same session id always maps to the same UUID.
_SESSION_NAMESPACE = uuid.UUID("6f1d3c4e-8a7b-4c2d-9e10-5a6b7c8d9e0f")

_connection: psycopg.Connection | None = None
_lock = threading.Lock()


def session_id_to_uuid(session_id: str) -> str:
    """Return ``session_id`` if it is already a UUID, else a deterministic UUIDv5."""
    try:
        return str(uuid.UUID(session_id))
    except ValueError:
        return str(uuid.uuid5(_SESSION_NAMESPACE, session_id))


def _get_connection() -> psycopg.Connection:
    """Shared connection, reopened if it was closed or broken.

    psycopg 3 connections are thread-safe (operations are serialized), which is
    sufficient for this app. Use a connection pool for high-concurrency deployments.
    """
    global _connection
    with _lock:
        if _connection is None or _connection.closed or _connection.broken:
            _connection = psycopg.connect(get_settings().psycopg_conninfo, autocommit=True)
        return _connection


def init_postgres_memory() -> None:
    table = get_settings().postgres_history_table
    PostgresChatMessageHistory.create_tables(_get_connection(), table)
    logger.info("postgres_memory_ready", extra={"table": table})


def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        get_settings().postgres_history_table,
        session_id_to_uuid(session_id),
        sync_connection=_get_connection(),
    )


def close_postgres_memory() -> None:
    global _connection
    with _lock:
        if _connection is not None and not _connection.closed:
            _connection.close()
        _connection = None
