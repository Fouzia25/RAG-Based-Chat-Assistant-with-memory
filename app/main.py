"""FastAPI entrypoint: ``uvicorn app.main:app --reload``."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.memory.factory import init_memory, shutdown_memory
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    logger.info(
        "startup",
        extra={
            "memory_backend": settings.memory_backend,
            "model": settings.openai_model,
            "collection": settings.collection_name,
            "openai_key_present": settings.openai_api_key is not None,
        },
    )
    try:
        init_memory()  # creates the Postgres history table if needed
    except Exception:
        # Keep serving so /health can report what's wrong instead of crash-looping.
        logger.exception("memory_init_failed", extra={"memory_backend": settings.memory_backend})
    yield
    shutdown_memory()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Chat Assistant",
        version=__version__,
        description="Chat with your documents: grounded answers, citations, and persistent memory.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    def _error(request: Request, status_code: int, error: str, detail: str | None = None) -> JSONResponse:
        return JSONResponse(
            {"error": error, "detail": detail, "request_id": getattr(request.state, "request_id", None)},
            status_code=status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return _error(request, 422, "validation_error", str(exc.errors()))

    @app.exception_handler(openai.AuthenticationError)
    async def openai_auth_error(request: Request, exc: openai.AuthenticationError):
        logger.error("openai_auth_failed")
        return _error(request, 502, "openai_authentication_failed", "Check OPENAI_API_KEY.")

    @app.exception_handler(openai.RateLimitError)
    async def openai_rate_limit(request: Request, exc: openai.RateLimitError):
        return _error(request, 429, "openai_rate_limited", "OpenAI rate limit or quota exceeded.")

    @app.exception_handler(openai.APIError)
    async def openai_error(request: Request, exc: openai.APIError):
        logger.exception("openai_error")
        return _error(request, 502, "openai_error", type(exc).__name__)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Request, exc: RuntimeError):
        # e.g. missing OPENAI_API_KEY raised by Settings.require_openai_key()
        logger.exception("runtime_error")
        return _error(request, 503, "service_unavailable", str(exc))

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("unhandled_error")
        return _error(request, 500, "internal_error", type(exc).__name__)

    app.include_router(router)
    return app


app = create_app()
