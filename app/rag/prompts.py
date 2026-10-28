"""Prompt templates for question rewriting and grounded answering."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

NO_ANSWER = "I don't know based on the provided documents."

CONDENSE_SYSTEM = (
    "Given the chat history and the latest user question, which may refer to earlier "
    "messages, rewrite it as a standalone question that can be understood without the "
    "history. Keep the user's language and intent. Do NOT answer the question; only "
    "return the rewritten question. If it is already standalone, return it unchanged."
)

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_SYSTEM = f"""You are a helpful assistant that answers questions using ONLY the context below.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge, and do not guess.
2. If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
3. Cite every fact with its source in the form [filename, page], e.g. [handbook.pdf, 3].
   Use only the filenames and pages shown in the context headers.
4. Be concise. Use bullet points for lists.
5. Use the chat history only to understand what the user is referring to, never as a source of facts.

Context:
{{context}}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
