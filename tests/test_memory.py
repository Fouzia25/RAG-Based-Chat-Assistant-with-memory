import uuid

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.memory import factory, postgres_memory, redis_memory
from app.memory.postgres_memory import session_id_to_uuid


def test_session_id_to_uuid_is_deterministic():
    assert session_id_to_uuid("alice") == session_id_to_uuid("alice")
    assert session_id_to_uuid("alice") != session_id_to_uuid("bob")
    uuid.UUID(session_id_to_uuid("alice"))  # valid UUID


def test_session_id_to_uuid_passes_through_real_uuids():
    sid = str(uuid.uuid4())
    assert session_id_to_uuid(sid) == sid


def test_factory_uses_postgres_backend(monkeypatch):
    calls = {}

    class FakePGHistory:
        def __init__(self, table, session_id, *, sync_connection):
            calls.update(table=table, session_id=session_id, conn=sync_connection)

    monkeypatch.setattr(postgres_memory, "PostgresChatMessageHistory", FakePGHistory)
    monkeypatch.setattr(postgres_memory, "_get_connection", lambda: "CONN")

    history = factory.get_session_history("user-42")

    assert isinstance(history, FakePGHistory)
    assert calls == {
        "table": "chat_history",
        "session_id": session_id_to_uuid("user-42"),
        "conn": "CONN",
    }


def test_factory_uses_redis_backend_with_ttl(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "redis")
    monkeypatch.setenv("REDIS_TTL_SECONDS", "3600")
    monkeypatch.setenv("REDIS_KEY_PREFIX", "test:")
    get_settings.cache_clear()
    calls = {}

    class FakeRedisHistory:
        def __init__(self, session_id, **kwargs):
            calls.update(session_id=session_id, **kwargs)

    monkeypatch.setattr(redis_memory, "RedisChatMessageHistory", FakeRedisHistory)
    monkeypatch.setattr(redis_memory, "get_redis_client", lambda: "CLIENT")

    history = factory.get_session_history("user-42")

    assert isinstance(history, FakeRedisHistory)
    assert calls == {"session_id": "user-42", "redis_client": "CLIENT", "key_prefix": "test:", "ttl": 3600}


def test_empty_ttl_means_no_expiry(monkeypatch):
    monkeypatch.setenv("REDIS_TTL_SECONDS", "")
    assert Settings().redis_ttl_seconds is None


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "mongodb")
    with pytest.raises(ValidationError):
        Settings()


def test_backend_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "Redis")
    assert Settings().memory_backend == "redis"


def test_init_memory_creates_postgres_table(monkeypatch):
    created = {}
    monkeypatch.setattr(postgres_memory, "_get_connection", lambda: "CONN")
    monkeypatch.setattr(
        postgres_memory.PostgresChatMessageHistory,
        "create_tables",
        staticmethod(lambda conn, table: created.update(conn=conn, table=table)),
    )

    factory.init_memory()

    assert created == {"conn": "CONN", "table": "chat_history"}
