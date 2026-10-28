"""Chooses the chat-history backend from ``MEMORY_BACKEND`` (postgres | redis)."""

from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory

from app.config import get_settings


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    backend = get_settings().memory_backend
    if backend == "postgres":
        from app.memory.postgres_memory import get_postgres_history

        return get_postgres_history(session_id)
    if backend == "redis":
        from app.memory.redis_memory import get_redis_history

        return get_redis_history(session_id)
    raise ValueError(f"Unsupported MEMORY_BACKEND: {backend!r}")


def init_memory() -> None:
    """Prepare the selected backend (creates the Postgres table if needed)."""
    backend = get_settings().memory_backend
    if backend == "postgres":
        from app.memory.postgres_memory import init_postgres_memory

        init_postgres_memory()
    elif backend == "redis":
        from app.memory.redis_memory import init_redis_memory

        init_redis_memory()


def shutdown_memory() -> None:
    if get_settings().memory_backend == "postgres":
        from app.memory.postgres_memory import close_postgres_memory

        close_postgres_memory()
