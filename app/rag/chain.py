"""The RAG chain (LCEL) plus per-session conversation memory.

Flow::

    {input, chat_history}
      -> rewrite question (only if there is history)
      -> retrieve docs -> format context + sources
      -> answer prompt -> ChatOpenAI -> str

Memory note: ``RunnableWithMessageHistory`` is deprecated as of
langchain-core 1.3.3 (removal planned for 2.0). ``ConversationalRAG`` does the
same thing explicitly: load history -> trim -> run chain -> save the new turn.
That keeps the chain itself stateless and works with any
``BaseChatMessageHistory`` (Postgres or Redis).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from operator import itemgetter
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.rag.prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT
from app.utils.logging import get_logger

logger = get_logger(__name__)

HistoryFactory = Callable[[str], BaseChatMessageHistory]


def get_llm(streaming: bool = False) -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        api_key=settings.require_openai_key(),
        timeout=settings.openai_timeout_seconds,
        max_retries=2,
        streaming=streaming,
    )


# --- formatting helpers -------------------------------------------------------


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no relevant documents were found)"
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """One entry per (file, page), in retrieval order."""
    seen: set[tuple[str, Any]] = set()
    sources = []
    for d in docs:
        key = (d.metadata.get("source", "unknown"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": key[0],
                "page": key[1],
                "chunk_index": d.metadata.get("chunk_index"),
                "snippet": d.page_content[:300],
            }
        )
    return sources


def build_history_trimmer(max_messages: int) -> Runnable:
    """Keep only the last N messages (counted with len), starting on a human turn."""
    return trim_messages(
        max_tokens=max_messages,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )


# --- chain construction -------------------------------------------------------


def build_rewrite_step(llm: BaseChatModel) -> Runnable:
    """History-aware query rewriting; skips the LLM call when there is no history."""
    condense = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    def route(inputs: dict) -> Any:
        if inputs.get("chat_history"):
            return condense  # a returned Runnable is invoked by LCEL
        return inputs["input"]

    return RunnableLambda(route).with_config(run_name="rewrite_question")


def build_retrieval_chain(llm: BaseChatModel, retriever: Runnable) -> Runnable:
    """{input, chat_history} -> + standalone_question, docs, context, sources."""
    return (
        RunnablePassthrough.assign(standalone_question=build_rewrite_step(llm))
        .assign(docs=itemgetter("standalone_question") | retriever)
        .assign(
            context=lambda x: format_docs(x["docs"]),
            sources=lambda x: extract_sources(x["docs"]),
        )
        .with_config(run_name="retrieve")
    )


def build_answer_chain(llm: BaseChatModel) -> Runnable:
    """{input, chat_history, context} -> answer string."""
    return (ANSWER_PROMPT | llm | StrOutputParser()).with_config(run_name="answer")


def build_rag_chain(llm: BaseChatModel, retriever: Runnable) -> Runnable:
    """Full stateless chain: returns a dict including `answer` and `sources`."""
    return build_retrieval_chain(llm, retriever).assign(answer=build_answer_chain(llm))


# --- memory-aware wrapper -----------------------------------------------------


@dataclass
class ChatResult:
    answer: str
    sources: list[dict[str, Any]]
    standalone_question: str


class ConversationalRAG:
    """Runs the RAG chain with per-session history loaded from / saved to a backend."""

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Runnable,
        get_session_history: HistoryFactory,
        max_history_messages: int = 10,
    ) -> None:
        self.get_session_history = get_session_history
        self.retrieval_chain = build_retrieval_chain(llm, retriever)
        self.answer_chain = build_answer_chain(llm)
        self.rag_chain = self.retrieval_chain.assign(answer=self.answer_chain)
        self.trimmer = build_history_trimmer(max_history_messages)
        self.max_history_messages = max_history_messages

    def _load_history(self, history: BaseChatMessageHistory) -> list[BaseMessage]:
        if self.max_history_messages == 0:
            return []
        return self.trimmer.invoke(history.messages)

    @staticmethod
    def _save_turn(history: BaseChatMessageHistory, question: str, answer: str) -> None:
        history.add_messages([HumanMessage(content=question), AIMessage(content=answer)])

    def invoke(self, session_id: str, question: str) -> ChatResult:
        history = self.get_session_history(session_id)
        inputs = {"input": question, "chat_history": self._load_history(history)}
        out = self.rag_chain.invoke(inputs, config={"metadata": {"session_id": session_id}})
        self._save_turn(history, question, out["answer"])
        return ChatResult(out["answer"], out["sources"], out["standalone_question"])

    def stream(self, session_id: str, question: str) -> Iterator[dict[str, Any]]:
        """Yield ``{"type": "sources"|"token"|"done", ...}`` events.

        Retrieval runs first so sources can be sent before the first token.
        The turn is saved to memory only after the answer completes.
        """
        history = self.get_session_history(session_id)
        config = {"metadata": {"session_id": session_id}}
        state = self.retrieval_chain.invoke(
            {"input": question, "chat_history": self._load_history(history)}, config=config
        )
        yield {
            "type": "sources",
            "sources": state["sources"],
            "standalone_question": state["standalone_question"],
        }

        parts: list[str] = []
        for token in self.answer_chain.stream(state, config=config):
            parts.append(token)
            yield {"type": "token", "text": token}

        answer = "".join(parts)
        self._save_turn(history, question, answer)
        yield {"type": "done", "answer": answer}


def build_conversational_rag() -> ConversationalRAG:
    """Production wiring: ChatOpenAI + PGVector retriever + configured memory backend."""
    from app.memory.factory import get_session_history
    from app.rag.retriever import get_retriever

    return ConversationalRAG(
        llm=get_llm(streaming=True),
        retriever=get_retriever(),
        get_session_history=get_session_history,
        max_history_messages=get_settings().max_history_messages,
    )
