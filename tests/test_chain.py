from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.rag.chain import build_history_trimmer, build_rag_chain, extract_sources, format_docs
from app.rag.prompts import ANSWER_SYSTEM, NO_ANSWER


def test_format_docs_includes_citation_headers(sample_docs):
    text = format_docs(sample_docs)
    assert "[handbook.md, 1]" in text
    assert "[faq.txt, 1]" in text
    assert format_docs([]) == "(no relevant documents were found)"


def test_extract_sources_dedupes_by_file_and_page(sample_docs):
    sources = extract_sources(sample_docs)
    assert [(s["source"], s["page"]) for s in sources] == [("handbook.md", 1), ("faq.txt", 1)]
    assert sources[0]["chunk_index"] == 0


def test_system_prompt_enforces_grounding():
    assert NO_ANSWER in ANSWER_SYSTEM
    assert "[filename, page]" in ANSWER_SYSTEM


def test_rag_chain_without_history_skips_rewrite(retriever):
    llm = FakeListChatModel(responses=["25 days [handbook.md, 1]"])
    chain = build_rag_chain(llm, retriever.runnable)

    out = chain.invoke({"input": "How much vacation?", "chat_history": []})

    assert out["answer"] == "25 days [handbook.md, 1]"
    assert out["standalone_question"] == "How much vacation?"
    assert retriever.queries == ["How much vacation?"]  # original question used as-is
    assert {s["source"] for s in out["sources"]} == {"handbook.md", "faq.txt"}


def test_rag_chain_rewrites_follow_up_with_history(retriever):
    llm = FakeListChatModel(responses=["How far in advance must vacation be requested?", "Two weeks."])
    chain = build_rag_chain(llm, retriever.runnable)
    history = [HumanMessage("How much vacation do I get?"), AIMessage("25 days.")]

    out = chain.invoke({"input": "And how early do I ask?", "chat_history": history})

    assert retriever.queries == ["How far in advance must vacation be requested?"]
    assert out["standalone_question"] == "How far in advance must vacation be requested?"
    assert out["answer"] == "Two weeks."


def test_conversational_rag_saves_turns(make_assistant, histories):
    assistant = make_assistant(["25 days.", "Rewritten question?", "Two weeks."])

    first = assistant.invoke("s1", "How much vacation?")
    second = assistant.invoke("s1", "How early do I ask?")

    assert first.answer == "25 days."
    assert second.standalone_question == "Rewritten question?"
    stored = histories["s1"].messages
    assert [m.type for m in stored] == ["human", "ai", "human", "ai"]
    assert stored[-1].content == "Two weeks."


def test_sessions_are_isolated(make_assistant, histories):
    assistant = make_assistant(["A", "B"])
    assistant.invoke("alice", "q1")
    assistant.invoke("bob", "q2")  # no history for bob -> no rewrite call
    assert len(histories["alice"].messages) == 2
    assert len(histories["bob"].messages) == 2


def test_stream_emits_sources_then_tokens_then_done(make_assistant, histories):
    assistant = make_assistant(["Hello world"])

    events = list(assistant.stream("s1", "hi"))

    assert events[0]["type"] == "sources"
    assert events[0]["sources"]
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "Hello world"
    assert len(tokens) > 1  # actually streamed, not one blob
    assert events[-1] == {"type": "done", "answer": "Hello world"}
    assert histories["s1"].messages[-1].content == "Hello world"


def test_history_trimmer_keeps_last_messages_starting_with_human():
    messages = []
    for i in range(6):
        messages += [HumanMessage(f"q{i}"), AIMessage(f"a{i}")]

    trimmed = build_history_trimmer(5).invoke(messages)

    # 5 would start on an AI message, so trim_messages drops it to start on a human turn.
    assert [m.content for m in trimmed] == ["q4", "a4", "q5", "a5"]


def test_zero_history_limit_sends_no_history(make_assistant, retriever):
    assistant = make_assistant(["A", "B"], max_history_messages=0)
    assistant.invoke("s1", "first")
    assistant.invoke("s1", "second")
    # No history -> never rewritten -> retriever sees raw questions.
    assert retriever.queries == ["first", "second"]
