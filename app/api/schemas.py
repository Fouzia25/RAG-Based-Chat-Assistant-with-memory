"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,128}$"


class ChatRequest(BaseModel):
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN, examples=["demo-session-1"])
    message: str = Field(..., min_length=1, max_length=8000, examples=["What is the refund policy?"])


class Source(BaseModel):
    source: str = Field(..., description="Original filename")
    page: int | None = Field(None, description="1-based page number")
    chunk_index: int | None = None
    snippet: str = Field("", description="Start of the retrieved chunk")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    standalone_question: str = Field(..., description="The question after history-aware rewriting")


class FileIngestResult(BaseModel):
    filename: str
    chunks_total: int
    chunks_added: int
    chunks_skipped: int = Field(..., description="Chunks already indexed (deduplicated)")
    error: str | None = None


class IngestResponse(BaseModel):
    files: list[FileIngestResult]
    chunks_added: int
    chunks_skipped: int


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system", "other"]
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    backend: str
    messages: list[HistoryMessage]


class DeleteResponse(BaseModel):
    session_id: str
    cleared: bool


class ComponentHealth(BaseModel):
    ok: bool
    required: bool = True
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    memory_backend: str
    checks: dict[str, ComponentHealth]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
