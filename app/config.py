"""Application configuration, loaded from environment variables and `.env`.

Every setting maps 1:1 to an upper-case environment variable
(e.g. ``chunk_size`` <- ``CHUNK_SIZE``). See ``.env.example`` for documentation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenAI -----------------------------------------------------------
    # SecretStr keeps the key out of repr()/logs; call .get_secret_value() to use it.
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- PostgreSQL (pgvector + optional chat memory) ----------------------
    # SQLAlchemy URL with the psycopg3 driver.
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    collection_name: str = "rag_documents"
    postgres_history_table: str = "chat_history"

    # --- Redis (optional chat memory) --------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "chat:"
    redis_ttl_seconds: int | None = Field(default=None, ge=1)

    # --- Memory -------------------------------------------------------------
    memory_backend: Literal["postgres", "redis"] = "postgres"
    max_history_messages: int = Field(default=10, ge=0)

    # --- Ingestion ----------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_mb: int = Field(default=20, gt=0)

    # --- Retrieval ----------------------------------------------------------
    search_type: Literal["similarity", "mmr"] = "similarity"
    retriever_k: int = Field(default=4, gt=0)
    mmr_fetch_k: int = Field(default=20, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- App ----------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = ["http://localhost:8501"]

    @field_validator("memory_backend", "search_type", mode="before")
    @classmethod
    def _lowercase(cls, value: str) -> str:
        return value.lower() if isinstance(value, str) else value

    @field_validator("redis_ttl_seconds", "score_threshold", "openai_api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, value):
        # Lets `.env` contain `REDIS_TTL_SECONDS=` to mean "unset".
        return None if value == "" else value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def psycopg_conninfo(self) -> str:
        """`database_url` in the plain libpq form that psycopg.connect() expects."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    def require_openai_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
        return self.openai_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
