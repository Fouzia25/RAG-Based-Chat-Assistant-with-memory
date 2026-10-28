"""HTTP routes.

Endpoints are plain ``def`` (not ``async def``): the LangChain chain, psycopg
and Redis calls here are synchronous, and FastAPI runs sync endpoints in a
thread pool so they don't block the event loop.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import BaseMessage

from app.api.schemas import (
    SESSION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ComponentHealth,
    DeleteResponse,
    FileIngestResult,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
    IngestResponse,
    Source,
)
from app.config import get_settings
from app.ingestion.ingest import ingest_files
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.memory.factory import get_session_history
from app.rag.chain import ConversationalRAG, HistoryFactory, build_conversational_rag
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# --- dependencies (overridden in tests) ------------------------------------------


@lru_cache
def _cached_assistant() -> ConversationalRAG:
    return build_conversational_rag()


def get_assistant() -> ConversationalRAG:
    return _cached_assistant()


def get_history_factory() -> HistoryFactory:
    return get_session_history


def get_ingest_fn():
    return ingest_files


# --- chat -------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest, assistant: ConversationalRAG = Depends(get_assistant)) -> ChatResponse:
    result = assistant.invoke(req.session_id, req.message)
    logger.info("chat_completed", extra={"session_id": req.session_id, "sources": len(result.sources)})
    return ChatResponse(
        session_id=req.session_id,
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        standalone_question=result.standalone_question,
    )


def _sse(event: str, data: dict) -> str:
    # JSON-encode the payload so newlines inside tokens can't break SSE framing.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest, assistant: ConversationalRAG = Depends(get_assistant)) -> StreamingResponse:
    """Server-Sent Events: `sources` (once) -> `token` (many) -> `done` | `error`."""

    def event_stream() -> Iterator[str]:
        try:
            for event in assistant.stream(req.session_id, req.message):
                kind = event.pop("type")
                yield _sse(kind, event)
            logger.info("chat_stream_completed", extra={"session_id": req.session_id})
        except Exception as exc:
            # Headers are already sent, so report the failure in-band.
            logger.exception("chat_stream_failed", extra={"session_id": req.session_id})
            yield _sse("error", {"error": type(exc).__name__, "detail": "The assistant failed to answer."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- sessions --------------------------------------------------------------------


def _role(message: BaseMessage) -> str:
    return {"human": "user", "ai": "assistant", "system": "system"}.get(message.type, "other")


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse, tags=["sessions"])
def get_history(
    session_id: str = PathParam(..., pattern=SESSION_ID_PATTERN),
    history_factory: HistoryFactory = Depends(get_history_factory),
) -> HistoryResponse:
    messages = history_factory(session_id).messages
    return HistoryResponse(
        session_id=session_id,
        backend=get_settings().memory_backend,
        messages=[HistoryMessage(role=_role(m), content=m.text) for m in messages],
    )


@router.delete("/sessions/{session_id}", response_model=DeleteResponse, tags=["sessions"])
def delete_session(
    session_id: str = PathParam(..., pattern=SESSION_ID_PATTERN),
    history_factory: HistoryFactory = Depends(get_history_factory),
) -> DeleteResponse:
    history_factory(session_id).clear()
    logger.info("session_cleared", extra={"session_id": session_id})
    return DeleteResponse(session_id=session_id, cleared=True)


# --- ingestion --------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest(files: list[UploadFile] = File(...), ingest_fn=Depends(get_ingest_fn)) -> IngestResponse:
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"'{f.filename}' is not supported. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        if f.size is not None and f.size > max_bytes:
            raise HTTPException(
                413,
                f"'{f.filename}' exceeds MAX_UPLOAD_MB={settings.max_upload_mb}",
            )

    with tempfile.TemporaryDirectory(prefix="rag-upload-") as tmp:
        saved: list[tuple[Path, str]] = []
        for i, f in enumerate(files):
            # Path(...).name strips any directory components (path traversal).
            display_name = Path(f.filename or f"upload-{i}").name
            dest = Path(tmp) / f"{i}_{display_name}"
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append((dest, display_name))
        result = ingest_fn(saved)

    return IngestResponse(
        files=[FileIngestResult(**vars(r)) for r in result.files],
        chunks_added=result.chunks_added,
        chunks_skipped=result.chunks_skipped,
    )


# --- health -----------------------------------------------------------------------


def check_openai_key() -> ComponentHealth:
    key = get_settings().openai_api_key
    present = key is not None and bool(key.get_secret_value())
    return ComponentHealth(ok=present, detail=None if present else "OPENAI_API_KEY is not set")


def check_postgres() -> ComponentHealth:
    import psycopg

    try:
        with psycopg.connect(get_settings().psycopg_conninfo, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return ComponentHealth(ok=True)
    except Exception as exc:
        return ComponentHealth(ok=False, detail=type(exc).__name__)


def check_redis() -> ComponentHealth:
    import redis

    required = get_settings().memory_backend == "redis"
    try:
        client = redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=3, socket_timeout=3)
        client.ping()
        client.close()
        return ComponentHealth(ok=True, required=required)
    except Exception as exc:
        return ComponentHealth(ok=False, required=required, detail=type(exc).__name__)


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> JSONResponse:
    checks = {"openai_api_key": check_openai_key(), "postgres": check_postgres(), "redis": check_redis()}
    healthy = all(c.ok for c in checks.values() if c.required)
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        memory_backend=get_settings().memory_backend,
        checks=checks,
    )
    return JSONResponse(body.model_dump(), status_code=200 if healthy else 503)
