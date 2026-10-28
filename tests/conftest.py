"""Shared fixtures. No test talks to OpenAI, Postgres or Redis."""

from __future__ import annotations

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app.config import get_settings
from app.rag.chain import ConversationalRAG


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Isolate tests from any real .env file and clear cached settings."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000")
    monkeypatch.setenv("MEMORY_BACKEND", "postgres")
    monkeypatch.setenv("LOG_JSON", "false")
    monkeypatch.chdir(__import__("pathlib").Path(__file__).parent)  # no .env here
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Employees get 25 days of paid vacation per year.",
            metadata={"source": "handbook.md", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="Vacation requests must be submitted two weeks in advance.",
            metadata={"source": "handbook.md", "page": 1, "chunk_index": 1},
        ),
        Document(
            page_content="Refunds are processed within 14 days.",
            metadata={"source": "faq.txt", "page": 1, "chunk_index": 0},
        ),
    ]


class RecordingRetriever:
    """Fake retriever that records the queries it receives."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.queries: list[str] = []
        self.runnable = RunnableLambda(self._retrieve)

    def _retrieve(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.docs


@pytest.fixture
def retriever(sample_docs) -> RecordingRetriever:
    return RecordingRetriever(sample_docs)


@pytest.fixture
def histories() -> dict[str, InMemoryChatMessageHistory]:
    return {}


@pytest.fixture
def make_assistant(retriever, histories):
    def factory(responses: list[str], max_history_messages: int = 10) -> ConversationalRAG:
        return ConversationalRAG(
            llm=FakeListChatModel(responses=responses),
            retriever=retriever.runnable,
            get_session_history=lambda sid: histories.setdefault(sid, InMemoryChatMessageHistory()),
            max_history_messages=max_history_messages,
        )

    return factory
