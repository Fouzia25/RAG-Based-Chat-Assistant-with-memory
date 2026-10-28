"""Streamlit chat UI for the RAG Chat Assistant.

Run:  streamlit run ui/streamlit_app.py
Talks to the FastAPI backend at API_URL (default http://localhost:8000).
The session id lives in the URL (?session=...), so a page refresh keeps the conversation.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = (5, 300)  # (connect, read) seconds

st.set_page_config(page_title="RAG chat assistant", page_icon=":material/forum:")


# --- backend helpers ------------------------------------------------------------


def fetch_history(session_id: str) -> list[dict]:
    try:
        resp = requests.get(f"{API_URL}/sessions/{session_id}/history", timeout=TIMEOUT)
        resp.raise_for_status()
        return [{"role": m["role"], "content": m["content"], "sources": []} for m in resp.json()["messages"]]
    except requests.RequestException:
        return []


def stream_answer(session_id: str, message: str, holder: dict) -> Iterator[str]:
    """Yield answer tokens from the SSE endpoint; put sources/errors in `holder`."""
    with requests.post(
        f"{API_URL}/chat/stream",
        json={"session_id": session_id, "message": message},
        stream=True,
        timeout=TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        event = None
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
                if event == "sources":
                    holder["sources"] = data["sources"]
                elif event == "token":
                    yield data["text"]
                elif event == "error":
                    holder["error"] = data.get("detail", "Unknown error")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/description:"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s.get('page') or '–'}")
            if s.get("snippet"):
                st.caption(s["snippet"])


def start_new_session() -> None:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.query_params["session"] = st.session_state.session_id


# --- session state ----------------------------------------------------------------

if "session_id" not in st.session_state:
    session_from_url = st.query_params.get("session")
    if session_from_url:
        st.session_state.session_id = session_from_url
        st.session_state.messages = fetch_history(session_from_url)
    else:
        start_new_session()

# --- sidebar ----------------------------------------------------------------------

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", icon=":material/upload:", disabled=not uploads, width="stretch"):
        with st.spinner("Embedding and indexing..."):
            try:
                files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
                resp = requests.post(f"{API_URL}/ingest", files=files, timeout=TIMEOUT)
                resp.raise_for_status()
                body = resp.json()
                st.success(f"{body['chunks_added']} chunks added, {body['chunks_skipped']} already indexed.")
                for f in body["files"]:
                    if f["error"]:
                        st.error(f"{f['filename']}: {f['error']}")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.header("Session")
    st.code(st.session_state.session_id, language=None)
    st.button("New session", icon=":material/add_comment:", on_click=start_new_session, width="stretch")

# --- chat -------------------------------------------------------------------------

st.title("RAG chat assistant")
st.caption("Answers come only from your indexed documents, with citations.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask a question about your documents", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        holder: dict = {"sources": []}
        try:
            answer = st.write_stream(stream_answer(st.session_state.session_id, prompt, holder))
        except requests.RequestException as exc:
            answer = ""
            holder["error"] = f"Could not reach the API at {API_URL}: {exc}"
        if holder.get("error"):
            st.error(holder["error"])
        render_sources(holder["sources"])

    if answer:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": holder["sources"]}
        )
