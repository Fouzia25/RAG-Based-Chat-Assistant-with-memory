import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.api import routes
from app.api.schemas import ComponentHealth
from app.ingestion.ingest import FileIngestionResult, IngestionResult
from app.main import create_app


@pytest.fixture
def client(make_assistant, histories):
    app = create_app()
    assistant = make_assistant(["25 days [handbook.md, 1]", "Rewritten?", "Two weeks [handbook.md, 1]"])
    app.dependency_overrides[routes.get_assistant] = lambda: assistant
    app.dependency_overrides[routes.get_history_factory] = lambda: (
        lambda sid: histories.setdefault(sid, InMemoryChatMessageHistory())
    )
    # Not used as a context manager, so the lifespan (DB init) does not run.
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_returns_answer_and_sources(client):
    resp = client.post("/chat", json={"session_id": "s1", "message": "How much vacation?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "25 days [handbook.md, 1]"
    assert body["sources"][0]["source"] == "handbook.md"
    assert body["standalone_question"] == "How much vacation?"
    assert "x-request-id" in resp.headers


def test_chat_validates_input(client):
    assert client.post("/chat", json={"session_id": "s1", "message": ""}).status_code == 422
    assert client.post("/chat", json={"session_id": "bad id!", "message": "hi"}).status_code == 422
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_chat_stream_sse(client):
    resp = client.post("/chat/stream", json={"session_id": "s1", "message": "How much vacation?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert tokens == "25 days [handbook.md, 1]"


def test_history_and_delete(client):
    client.post("/chat", json={"session_id": "s2", "message": "How much vacation?"})

    hist = client.get("/sessions/s2/history").json()
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant"]
    assert hist["messages"][0]["content"] == "How much vacation?"

    assert client.delete("/sessions/s2").json() == {"session_id": "s2", "cleared": True}
    assert client.get("/sessions/s2/history").json()["messages"] == []


def test_follow_up_uses_memory(client):
    client.post("/chat", json={"session_id": "s3", "message": "How much vacation?"})
    body = client.post("/chat", json={"session_id": "s3", "message": "How early do I ask?"}).json()
    assert body["standalone_question"] == "Rewritten?"
    assert body["answer"] == "Two weeks [handbook.md, 1]"


def test_ingest_upload(client):
    received = {}

    def fake_ingest(files):
        received["names"] = [name for _, name in files]
        received["content"] = files[0][0].read_text()
        return IngestionResult(files=[FileIngestionResult("notes.txt", 3, 3, 0)])

    client.app.dependency_overrides[routes.get_ingest_fn] = lambda: fake_ingest
    resp = client.post("/ingest", files=[("files", ("../../notes.txt", b"hello", "text/plain"))])

    assert resp.status_code == 200
    assert resp.json()["chunks_added"] == 3
    assert received == {"names": ["notes.txt"], "content": "hello"}  # path components stripped


def test_ingest_rejects_unsupported_type(client):
    resp = client.post("/ingest", files=[("files", ("photo.png", b"x", "image/png"))])
    assert resp.status_code == 400


def test_health_ok(client, monkeypatch):
    monkeypatch.setattr(routes, "check_postgres", lambda: ComponentHealth(ok=True))
    monkeypatch.setattr(routes, "check_redis", lambda: ComponentHealth(ok=False, required=False, detail="down"))
    resp = client.get("/health")
    assert resp.status_code == 200  # redis isn't required with the postgres backend
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["openai_api_key"]["ok"] is True


def test_health_degraded_without_key(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    routes.get_settings.cache_clear()
    monkeypatch.setattr(routes, "check_postgres", lambda: ComponentHealth(ok=True))
    monkeypatch.setattr(routes, "check_redis", lambda: ComponentHealth(ok=True, required=False))
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert "sk-" not in resp.text


def test_stream_error_is_reported_in_band(client):
    class Boom:
        def stream(self, *_):
            raise RuntimeError("db down")
            yield  # pragma: no cover

    client.app.dependency_overrides[routes.get_assistant] = lambda: Boom()
    resp = client.post("/chat/stream", json={"session_id": "s1", "message": "hi"})
    assert parse_sse(resp.text)[-1][0] == "error"
