"""Redis-backed chat history (``langchain_redis.RedisChatMessageHistory``).

Each message is stored as a RedisJSON document under
``{REDIS_KEY_PREFIX}{session_id}:{message_ulid}`` and indexed with the Redis
Query Engine, so the server needs the JSON + Search modules (Redis 8+ or
Redis Stack). If ``REDIS_TTL_SECONDS`` is set, every message key gets that TTL.
"""

from __future__ import annotations

from functools import lru_cache

import redis
from langchain_redis import RedisChatMessageHistory

from app.config import get_settings


@lru_cache
def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url)


def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    settings = get_settings()
    return RedisChatMessageHistory(
        session_id,
        redis_client=get_redis_client(),
        key_prefix=settings.redis_key_prefix,
        ttl=settings.redis_ttl_seconds,
    )


def init_redis_memory() -> None:
    # Fails fast on startup if Redis is unreachable.
    get_redis_client().ping()
