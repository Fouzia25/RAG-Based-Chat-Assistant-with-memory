"""Vector store and retriever over PostgreSQL + pgvector.

History-aware query rewriting happens in ``app.rag.chain`` (``build_rewrite_step``)
right before this retriever is called.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.require_openai_key(),
    )


@lru_cache
def get_vectorstore() -> PGVector:
    """PGVector store; creates the `vector` extension and its tables on first use."""
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
    )


def build_search_kwargs(settings: Settings) -> tuple[str, dict]:
    """Map our config onto LangChain's ``as_retriever`` search_type/search_kwargs."""
    if settings.search_type == "mmr":
        if settings.score_threshold is not None:
            logger.warning("score_threshold is ignored when SEARCH_TYPE=mmr")
        return "mmr", {
            "k": settings.retriever_k,
            "fetch_k": max(settings.mmr_fetch_k, settings.retriever_k),
            "lambda_mult": settings.mmr_lambda,
        }
    if settings.score_threshold is not None:
        # Relevance scores are normalized to [0, 1] (cosine: 1 - distance).
        return "similarity_score_threshold", {
            "k": settings.retriever_k,
            "score_threshold": settings.score_threshold,
        }
    return "similarity", {"k": settings.retriever_k}


def get_retriever(vectorstore: VectorStore | None = None) -> VectorStoreRetriever:
    search_type, search_kwargs = build_search_kwargs(get_settings())
    store = vectorstore or get_vectorstore()
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
