# RAG Chat Assistant with Memory

A chat assistant that answers questions **about your own documents**: PDFs,
Word files, text and Markdown. Every answer is:

- **Grounded**: it only uses facts found in your documents. If the answer isn't there, it says *"I don't know based on the provided documents."* and doesn't make something up.
- **Cited**: each fact is tagged with the file and page it came from, e.g. `[product_faq.txt, 1]`.
- **Conversational**: it remembers earlier turns, so follow-ups like *"and what about the team one?"* work.
- **Persistent**: the conversation is stored in PostgreSQL or Redis, so it survives restarts.

It ships as a **FastAPI** backend, a **Streamlit** chat UI, and a **Docker Compose**
setup that runs everything with one command.

---

## Table of contents

1. [Key concepts (what is RAG?)](#1-key-concepts-what-is-rag)
2. [Tech stack and why each piece is used](#2-tech-stack-and-why-each-piece-is-used)
3. [Architecture](#3-architecture)
4. [How ingestion works, step by step](#4-how-ingestion-works-step-by-step)
5. [How a chat request works, step by step](#5-how-a-chat-request-works-step-by-step)
6. [Conversation memory](#6-conversation-memory)
7. [Project structure, file by file](#7-project-structure-file-by-file)
8. [Getting started](#8-getting-started)
9. [Using the Streamlit UI](#9-using-the-streamlit-ui)
10. [API reference](#10-api-reference)
11. [Configuration reference](#11-configuration-reference)
12. [Testing](#12-testing)
13. [Logging, errors and security](#13-logging-errors-and-security)
14. [Troubleshooting](#14-troubleshooting)
15. [Deprecation notes](#15-deprecation-notes)
16. [Ideas for extending the project](#16-ideas-for-extending-the-project)

---

## 1. Key concepts (what is RAG?)

A large language model (LLM) like GPT only knows what it saw during training. It
doesn't know your company handbook or product FAQ, and if you ask about them it may
**hallucinate**, giving a confident answer that is made up.

**Retrieval-Augmented Generation (RAG)** fixes this in two phases:

| Phase | When | What happens |
|---|---|---|
| **Indexing** | Once per document | Split each document into small *chunks*, turn every chunk into an **embedding** (a list of numbers that captures its meaning) and store them in a **vector database**. |
| **Retrieval + generation** | Every question | Embed the question, find the chunks whose embeddings are **closest** in meaning, paste those chunks into the prompt as *context*, and tell the LLM to answer **only** from that context. |

Terms used throughout this README:

- **Chunk**: a piece of a document of up to `CHUNK_SIZE` (1000) characters. Small pieces make retrieval precise and keep prompts cheap.
- **Chunk overlap**: neighbouring chunks share `CHUNK_OVERLAP` (200) characters, so a sentence cut at a boundary still appears whole in one of them.
- **Embedding**: a vector (1536 numbers for `text-embedding-3-small`). Texts with similar meaning get similar vectors, so *"refund policy"* matches *"money back within 14 days"* even with no shared words.
- **Vector store**: a database that can find the nearest vectors quickly. This project uses **PostgreSQL with the pgvector extension**.
- **Similarity search**: return the `k` chunks closest to the question (cosine distance).
- **MMR (Maximal Marginal Relevance)**: return chunks that are relevant *and* different from each other, to avoid four near-identical chunks.
- **Standalone question**: a follow-up rewritten so it makes sense alone. *"Can I get a refund on it?"* becomes *"Can I get a refund on the Team plan?"* This is what makes retrieval work in a conversation.
- **Memory / chat history**: the stored list of previous user and assistant messages for one **session** (one conversation).

---

## 2. Tech stack and why each piece is used

| Component | Technology | Why |
|---|---|---|
| Orchestration | **LangChain** (LCEL: LangChain Expression Language) | Composable building blocks (prompts, retrievers, LLMs, history) joined with `\|` into a pipeline |
| LLM | **OpenAI `gpt-4o-mini`** | Cheap, fast, good at following "answer only from context" instructions |
| Embeddings | **OpenAI `text-embedding-3-small`** | Low cost, strong quality, 1536 dimensions |
| Vector store | **PostgreSQL 17 + pgvector** via `langchain-postgres` (`PGVector`) | One normal database for vectors *and* chat history; easy to run and back up |
| Memory (option A) | **PostgreSQL** via `PostgresChatMessageHistory` | Durable, no extra service needed |
| Memory (option B) | **Redis 8** via `RedisChatMessageHistory` | Very fast; supports automatic expiry (TTL) of old conversations |
| Document loading | `langchain-community` loaders + `pypdf`, `docx2txt` | PDF, DOCX, TXT and MD support |
| Chunking | `RecursiveCharacterTextSplitter` | Splits on paragraphs, then sentences, then words, keeping text natural |
| Backend | **FastAPI** + Uvicorn | Typed request validation, auto-generated docs at `/docs`, streaming responses |
| Frontend | **Streamlit** | A chat UI in about 150 lines of Python |
| Config | `pydantic-settings` | Reads `.env`, validates types and ranges, hides the API key |
| Packaging | **Docker** + Docker Compose | One command starts Postgres, Redis, API and UI |
| Tests | `pytest` + FastAPI `TestClient` | 39 tests that run without an API key or databases |

---

## 3. Architecture

```
 ┌────────────────────┐      HTTP / SSE      ┌──────────────────────────────────────────┐
 │  Streamlit UI      │ ───────────────────▶ │  FastAPI backend  (app/main.py)          │
 │  ui/streamlit_app  │ ◀─────────────────── │                                          │
 │  :8501             │                      │  /chat  /chat/stream  /ingest  /health   │
 └────────────────────┘                      │  /sessions/{id}/history                  │
        ▲                                    └───────┬───────────────┬──────────────────┘
        │ browser                                    │               │
   ┌────┴────┐                       ┌───────────────▼───┐   ┌───────▼──────────────────┐
   │  User   │                       │ ConversationalRAG │   │ Ingestion pipeline       │
   └─────────┘                       │ app/rag/chain.py  │   │ app/ingestion/ingest.py  │
                                     └──┬─────────┬──────┘   └───────┬──────────────────┘
                     load / save history│         │ retrieve         │ embed + store
                  ┌─────────────────────▼──┐   ┌──▼──────────────────▼───┐
                  │ Memory backend         │   │ PostgreSQL + pgvector   │
                  │ Postgres  OR  Redis    │   │ (document chunks)       │
                  └────────────────────────┘   └─────────────────────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │ OpenAI API         │
                           │ chat + embeddings  │
                           └────────────────────┘
```

In Docker Compose there are four containers:

| Service | Image | Port on your machine | Role |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | 5432 | Vector store, plus chat history when `MEMORY_BACKEND=postgres` |
| `redis` | `redis:8` | 6379 | Chat history when `MEMORY_BACKEND=redis` |
| `api` | built from `Dockerfile` | 8000 | FastAPI backend |
| `ui` | built from `Dockerfile` | 8501 | Streamlit frontend |

Inside Compose the containers reach each other by service name
(`postgres:5432`, `redis:6379`, `api:8000`). Data lives in the named volumes
`pgdata` and `redisdata`, so it survives `docker compose down`.

---

## 4. How ingestion works, step by step

Ingestion turns files into searchable chunks. You can trigger it two ways:

- **CLI**: `python -m app.ingestion.ingest --path ./data` (a file or a folder, searched recursively)
- **API / UI**: `POST /ingest` with uploaded files (the Streamlit sidebar uses this)

Both run the same pipeline in `app/ingestion/`:

```
file ──▶ 1. load ──▶ 2. normalize metadata ──▶ 3. split into chunks ──▶ 4. enrich metadata
     ──▶ 5. compute deterministic id ──▶ 6. skip already-stored ids ──▶ 7. embed + store new chunks
```

1. **Load** (`loaders.py`). The extension picks the loader:
   - `.pdf` → `PyPDFLoader`, one document per page
   - `.docx` → `Docx2txtLoader`
   - `.txt` / `.md` → `TextLoader` (Markdown is kept as plain text; headings stay readable for the LLM)
   - anything else → `UnsupportedFileTypeError` (HTTP 400 for uploads)
2. **Normalize metadata.** Each page gets `source` (filename), `page` (**1-based**, because PyPDF counts from 0) and `file_type`. Empty pages, such as image-only PDF pages, are skipped. TXT/MD/DOCX have no real pages, so they are reported as page 1.
3. **Split.** `RecursiveCharacterTextSplitter` cuts pages into chunks of up to `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap.
4. **Enrich.** Each chunk gets `chunk_index` (its position within the file), `content_hash` (SHA-256 of its text) and `upload_timestamp` (UTC).
5. **Deterministic id.** `id = sha256(source | page | content_hash)`. The same chunk of the same file always gets the same id.
6. **Deduplicate.** The pipeline asks the vector store which ids already exist and drops them. Duplicates within one batch are also collapsed. **Re-uploading a file costs nothing**: no embedding calls are made for chunks that are already stored.
7. **Embed and store.** Only new chunks are sent to OpenAI for embeddings and written to PGVector.

**One bad file doesn't abort the batch.** Its error is reported in the result and
the other files still get indexed.

Example output from the bundled sample docs:

```
  company_handbook.md: 2 added, 0 skipped
  product_faq.txt: 2 added, 0 skipped
Done. 4 chunks added, 0 already indexed (collection 'rag_documents').
```

Running it again prints `0 added, 2 skipped` for each file.

**Where are the vectors stored?** `langchain-postgres` creates two tables:
`langchain_pg_collection` (one row per collection, here `rag_documents`) and
`langchain_pg_embedding` (one row per chunk: id, text, vector and JSONB metadata).

> **Important:** if you change `OPENAI_EMBEDDING_MODEL`, `CHUNK_SIZE` or
> `CHUNK_OVERLAP`, re-ingest your documents, preferably into a new
> `COLLECTION_NAME`. Vectors from different models can't be compared.

---

## 5. How a chat request works, step by step

The core is `ConversationalRAG` in `app/rag/chain.py`. For a request
`{"session_id": "demo-1", "message": "..."}`:

```
 1. Load history ─▶ 2. Trim history ─▶ 3. Rewrite question ─▶ 4. Retrieve chunks
 ─▶ 5. Format context + sources ─▶ 6. Generate answer ─▶ 7. Save the turn ─▶ 8. Respond
```

1. **Load history.** Fetch every stored message for `session_id` from Postgres or Redis.
2. **Trim history.** Keep only the last `MAX_HISTORY_MESSAGES` messages (default 10, the last 5 question/answer pairs), always starting on a user message. This caps cost: long conversations don't grow the prompt forever. The full history stays in the database.
3. **Rewrite the question** (only if there is history). The LLM gets the history and the new message and is asked, via `CONDENSE_QUESTION_PROMPT`, to return a **standalone question**. With no history, this step is skipped and no LLM call is made.
4. **Retrieve.** The standalone question is embedded and the `RETRIEVER_K` closest chunks are fetched from PGVector (`similarity`, `mmr`, or similarity with a score threshold; see `app/rag/retriever.py`).
5. **Format.** Each chunk gets a header like `[product_faq.txt, 1]` so the LLM knows what to cite. Sources are de-duplicated by (file, page) for the response.
6. **Generate.** `ANSWER_PROMPT` sends the system rules, the context, the trimmed history and the user's message to `gpt-4o-mini` at temperature 0. The rules are:
   1. Use only facts in the context; no outside knowledge, no guessing.
   2. If the answer isn't in the context, reply exactly *"I don't know based on the provided documents."*
   3. Cite every fact as `[filename, page]`.
   4. Be concise; use bullet points for lists.
   5. Use the chat history only to understand references, never as a source of facts.
7. **Save the turn.** The user message and the assistant answer are appended to the session's history. When streaming, this happens only after the full answer has arrived.
8. **Respond** with the answer, the sources and the standalone question.

### A real example (from a live run on the sample docs)

| # | User asked | Standalone question (step 3) | Answer |
|---|---|---|---|
| 1 | *How much does the Pro plan cost?* | *(no history, unchanged)* | The Pro plan costs USD 6 per month `[product_faq.txt, 1]` |
| 2 | *And what about the team one? Can I get a refund on it?* | *What is the cost of the Team plan, and is it possible to get a refund for it?* | USD 12 per user per month; full refund within 14 days, prorated after that for annual plans only `[product_faq.txt, 1]` |
| 3 | *What is the capital of Australia?* | *(unchanged)* | *I don't know based on the provided documents.* |

Turn 2 shows **memory**: "the team one" and "it" were resolved from turn 1.
Turn 3 shows **grounding**: the model knows the answer from training but refuses
because it isn't in the documents.

### Streaming

`POST /chat/stream` runs the same steps but sends **Server-Sent Events** so the
UI can show the answer as it is typed:

```
event: sources   ← once, before the answer (retrieval runs first)
data: {"sources": [...], "standalone_question": "..."}

event: token     ← many times, one piece of text each
data: {"text": "The Pro plan"}

event: done      ← once, with the full answer
data: {"answer": "..."}
```

If something fails mid-stream, the HTTP status has already been sent, so the
error arrives as an `event: error` instead.

---

## 6. Conversation memory

Memory is chosen with one setting, `MEMORY_BACKEND`. The rest of the code only
sees LangChain's `BaseChatMessageHistory` interface (`.messages`,
`.add_messages()`, `.clear()`), so the two backends are interchangeable.
`app/memory/factory.py` picks the right one.

|  | `postgres` (default) | `redis` |
|---|---|---|
| Class | `PostgresChatMessageHistory` | `RedisChatMessageHistory` |
| Storage | Table `chat_history` (`id`, `session_id UUID`, `message JSONB`, `created_at`), created automatically at startup | One RedisJSON document per message, key `chat:<session_id>:<message_id>` |
| Requirements | The same Postgres used for vectors | Redis 8+ or Redis Stack (needs the JSON + Search modules) |
| Expiry | Never (delete with `DELETE /sessions/{id}`) | Optional: `REDIS_TTL_SECONDS` sets a time-to-live per message |
| Best for | Simplicity and durability | Speed, and conversations that should expire on their own |

**Session ids.** A session id is any string of letters, digits and `_ - : .`, up
to 128 characters. Postgres history requires a UUID, so non-UUID ids such as
`alice-1` are converted to a **stable UUIDv5** (`session_id_to_uuid` in
`postgres_memory.py`). The same id always maps to the same conversation.

**Isolation.** Sessions never see each other's history. Clearing one session
leaves the others untouched.

**Persistence.** History is in the database, not in the API process, so you can
restart or redeploy the API and conversations continue. This was verified in a
live run: after `docker compose restart api`, all 6 stored messages were still there.

**Switching to Redis:** set `MEMORY_BACKEND=redis` in `.env` and restart the API.
Existing Postgres history isn't copied over.

---

## 7. Project structure, file by file

```
rag-chat-assistant/
├── app/
│   ├── __init__.py            version string
│   ├── main.py                FastAPI app: startup, middleware, error handlers
│   ├── config.py              all settings (reads env vars / .env, validates them)
│   ├── api/
│   │   ├── routes.py          the HTTP endpoints
│   │   └── schemas.py         Pydantic request/response models
│   ├── rag/
│   │   ├── chain.py           the RAG pipeline + ConversationalRAG (memory wrapper)
│   │   ├── retriever.py       embeddings, PGVector store, search settings
│   │   └── prompts.py         rewrite prompt and grounded-answer prompt
│   ├── memory/
│   │   ├── factory.py         picks Postgres or Redis from MEMORY_BACKEND
│   │   ├── postgres_memory.py Postgres chat history (+ UUIDv5 mapping)
│   │   └── redis_memory.py    Redis chat history (+ optional TTL)
│   ├── ingestion/
│   │   ├── loaders.py         PDF/DOCX/TXT/MD loading, metadata normalization
│   │   └── ingest.py          split → dedupe → embed → store; CLI entry point
│   └── utils/
│       └── logging.py         JSON / text logging with API-key redaction
├── ui/
│   └── streamlit_app.py       chat UI (streaming, sources, upload, sessions)
├── data/sample_docs/          two fictional sample documents to try it out
│   ├── company_handbook.md
│   └── product_faq.txt
├── docker/init.sql            enables the pgvector extension on first DB start
├── tests/                     pytest suite (no API key or services needed)
├── Dockerfile                 Python 3.12 image, runs as a non-root user
├── docker-compose.yml         postgres + redis + api + ui
├── Makefile                   shortcuts (make up, make test, ...)
├── .env.example               every setting, documented; copy to .env
├── requirements.txt           pinned runtime dependencies
├── requirements-dev.txt       + pytest, httpx
└── pyproject.toml             project metadata and pytest settings
```

What the main modules do:

- **`app/config.py`**: a `Settings` class (pydantic-settings). Every field maps to an upper-case env var (`chunk_size` ← `CHUNK_SIZE`). It validates values (e.g. `CHUNK_OVERLAP` must be less than `CHUNK_SIZE`, `MEMORY_BACKEND` must be `postgres` or `redis`), treats empty values like `REDIS_TTL_SECONDS=` as unset, and stores the API key as a `SecretStr` so it never shows up in logs or reprs. `get_settings()` is cached, so the file is read once.
- **`app/main.py`**: creates the FastAPI app. On startup it sets up logging and creates the memory table. If the database is down it logs the error and keeps running, so `/health` can report the problem instead of the container crash-looping. Adds CORS for the UI, a request-logging middleware that assigns an `x-request-id`, and error handlers that turn exceptions into clean JSON (see [section 13](#13-logging-errors-and-security)).
- **`app/api/routes.py`**: the endpoints. They are plain `def` (not `async`) because LangChain, psycopg and Redis calls here are blocking; FastAPI runs them in a thread pool so the server stays responsive. Dependencies (`get_assistant`, `get_history_factory`, `get_ingest_fn`) are injected so tests can swap in fakes.
- **`app/rag/chain.py`**: builds the pipeline from small LCEL pieces (`build_rewrite_step`, `build_retrieval_chain`, `build_answer_chain`). The chain itself is **stateless**; `ConversationalRAG` adds memory around it (load → trim → run → save) and provides `invoke()` and `stream()`.
- **`app/rag/retriever.py`**: creates the cached `OpenAIEmbeddings` and `PGVector` objects and maps `SEARCH_TYPE` / `RETRIEVER_K` / `SCORE_THRESHOLD` / MMR settings onto LangChain's retriever options.
- **`app/rag/prompts.py`**: the two prompts described in [section 5](#5-how-a-chat-request-works-step-by-step).
- **`ui/streamlit_app.py`**: talks to the API only over HTTP and holds no LangChain code. It reads the SSE stream, renders tokens live and shows sources in an expandable panel.

---

## 8. Getting started

### Prerequisites

- An **OpenAI API key** (<https://platform.openai.com/api-keys>) with credit
- **Docker Desktop** (recommended path), or Python **3.11–3.13** for running locally

### Option A: Docker (recommended)

```bash
cd rag-chat-assistant
cp .env.example .env            # Windows PowerShell: Copy-Item .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

make up                         # or: docker compose up -d --build
make ingest-docker              # index data/sample_docs
```

Then open:

- Chat UI: <http://localhost:8501>
- Interactive API docs (Swagger): <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Useful commands:

| Command | What it does |
|---|---|
| `make logs` | Follow the API logs |
| `make down` | Stop containers, **keep** data |
| `make clean` | Stop containers and **delete** all data (vectors and history) |

No `make` on Windows? Every target is a single command; open the `Makefile` and
run that command directly, e.g.
`docker compose exec api python -m app.ingestion.ingest --path ./data/sample_docs`.

To index your own documents, upload them in the UI sidebar, or copy them into
`data/` and run `make ingest` (local) or use `POST /ingest`.

### Option B: Run locally (Python)

You still need Postgres (with pgvector) and, if you use it, Redis. The easiest
way is to start just those two with Docker:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                     # pip install -r requirements-dev.txt

docker compose up -d postgres redis
make ingest                      # python -m app.ingestion.ingest --path ./data
make run                         # API with auto-reload on :8000
make ui                          # in a second terminal: Streamlit on :8501
```

`.env` already uses `localhost` for `DATABASE_URL`, `REDIS_URL` and `API_URL`,
which is correct for local runs. Docker Compose overrides them with service names.

---

## 9. Using the Streamlit UI

- **Ask questions** in the chat box at the bottom. Answers stream in word by word.
- **Sources**: under each answer, expand *Sources (n)* to see the file, page and a snippet of each retrieved chunk.
- **Upload documents**: in the sidebar, pick PDF/DOCX/TXT/MD files and click *Index documents*. You'll see how many chunks were added and how many were already indexed.
- **Sessions**: the sidebar shows the current session id. It is also in the URL (`?session=...`), so **refreshing the page keeps the conversation** (history is reloaded from the API), and you can bookmark or share a conversation. *New session* starts a fresh conversation with empty memory.

---

## 10. API reference

Full interactive docs are at `/docs`. Summary:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Ask a question, get the full answer as JSON |
| `POST` | `/chat/stream` | Ask a question, get Server-Sent Events |
| `GET` | `/sessions/{session_id}/history` | Read a session's stored messages |
| `DELETE` | `/sessions/{session_id}` | Clear a session's memory |
| `POST` | `/ingest` | Upload and index files |
| `GET` | `/health` | Check that dependencies are reachable |

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "How much does the Pro plan cost?"}'
```

Request rules: `session_id` matches `^[A-Za-z0-9_\-:.]{1,128}$`; `message` is
1–8000 characters. Violations return `422`.

```json
{
  "session_id": "demo-1",
  "answer": "- The Pro plan costs USD 6 per month [product_faq.txt, 1].",
  "sources": [
    {
      "source": "product_faq.txt",
      "page": 1,
      "chunk_index": 0,
      "snippet": "Brightleaf Notes - Product FAQ (fictional sample)..."
    }
  ],
  "standalone_question": "How much does the Pro plan cost?"
}
```

### `POST /chat/stream`

Same request body. Response is `text/event-stream` with `sources`, `token`,
`done` and `error` events (see [Streaming](#streaming)). Use `curl -N` to watch it.

### `GET /sessions/{session_id}/history`

```json
{
  "session_id": "demo-1",
  "backend": "postgres",
  "messages": [
    {"role": "user", "content": "How much does the Pro plan cost?"},
    {"role": "assistant", "content": "- The Pro plan costs USD 6 per month [product_faq.txt, 1]."}
  ]
}
```

### `DELETE /sessions/{session_id}`

Returns `{"session_id": "demo-1", "cleared": true}`.

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@./handbook.pdf" -F "files=@./notes.docx"
```

```json
{
  "files": [
    {"filename": "handbook.pdf", "chunks_total": 12, "chunks_added": 12, "chunks_skipped": 0, "error": null}
  ],
  "chunks_added": 12,
  "chunks_skipped": 0
}
```

Errors: `400` for an unsupported extension, `413` for a file larger than
`MAX_UPLOAD_MB`. Uploaded files go into a temporary folder that is deleted
afterwards, and filenames are stripped of directory parts to prevent path
traversal.

### `GET /health`

```json
{
  "status": "ok",
  "memory_backend": "postgres",
  "checks": {
    "openai_api_key": {"ok": true, "required": true,  "detail": null},
    "postgres":       {"ok": true, "required": true,  "detail": null},
    "redis":          {"ok": true, "required": false, "detail": null}
  }
}
```

Returns `200` when all **required** checks pass, otherwise `503` with
`"status": "degraded"`. Redis is only required when `MEMORY_BACKEND=redis`.

---

## 11. Configuration reference

All settings are environment variables, usually set in `.env`. `.env.example`
documents each one inline. Changes take effect after restarting the API.

**OpenAI**

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | **Required.** Your secret key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for rewriting and answering |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (re-ingest if changed) |
| `OPENAI_TEMPERATURE` | `0` | 0 = deterministic, best for factual RAG (range 0–2) |
| `OPENAI_TIMEOUT_SECONDS` | `60` | Abort slow OpenAI requests |

**PostgreSQL**

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | Connection URL (host `postgres` inside Compose) |
| `COLLECTION_NAME` | `rag_documents` | PGVector collection for chunks; use different names to keep separate document sets |
| `POSTGRES_HISTORY_TABLE` | `chat_history` | Table for chat history |

**Redis**

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Connection URL (host `redis` inside Compose) |
| `REDIS_KEY_PREFIX` | `chat:` | Key prefix for history entries |
| `REDIS_TTL_SECONDS` | *(empty = never)* | Seconds before each stored message expires |

**Memory**

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_BACKEND` | `postgres` | `postgres` or `redis` |
| `MAX_HISTORY_MESSAGES` | `10` | Past messages sent to the LLM per request; `0` disables memory in prompts |

**Ingestion**

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Characters shared by neighbouring chunks (must be < `CHUNK_SIZE`) |
| `MAX_UPLOAD_MB` | `20` | Max size per uploaded file |

**Retrieval**

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_TYPE` | `similarity` | `similarity` (closest) or `mmr` (closest but diverse) |
| `RETRIEVER_K` | `4` | Chunks passed to the LLM as context |
| `MMR_FETCH_K` | `20` | MMR only: candidates fetched before re-ranking |
| `MMR_LAMBDA` | `0.5` | MMR only: 0 = most diverse, 1 = most relevant |
| `SCORE_THRESHOLD` | *(empty = off)* | Drop chunks with relevance below this (0–1). Similarity only. `0.3`–`0.5` is a good start |

> **Tip:** with the default settings the app always passes the top `RETRIEVER_K`
> chunks, even weakly related ones, so an unrelated file can appear in
> *Sources*. Setting `SCORE_THRESHOLD=0.3` keeps only relevant chunks.

**App / UI**

| Variable | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |
| `LOG_JSON` | `true` | One JSON object per log line (`false` = readable text) |
| `CORS_ORIGINS` | `["http://localhost:8501"]` | Browser origins allowed to call the API (JSON list) |
| `API_URL` | `http://localhost:8000` | Where the Streamlit UI finds the API |

---

## 12. Testing

```bash
make test          # locally:  python -m pytest
make test-docker   # inside the api container
```

The suite has **39 tests** and runs in a few seconds. It uses a **fake LLM**, a
**fake retriever / vector store** and **in-memory chat history**, so it needs
no OpenAI key, no database and no network, and costs nothing.

| File | What it covers |
|---|---|
| `tests/test_ingestion.py` | Loading and metadata, unsupported types, file discovery, chunk metadata, deterministic ids, re-ingest dedup, in-batch dedup, one bad file not stopping the batch |
| `tests/test_chain.py` | Context formatting with citation headers, source dedup, grounding rules in the prompt, rewrite skipped without history and used with it, turns saved, session isolation, stream event order, history trimming, `MAX_HISTORY_MESSAGES=0` |
| `tests/test_memory.py` | UUIDv5 mapping, backend selection (Postgres / Redis with TTL), empty TTL, invalid / case-insensitive backend names, table creation |
| `tests/test_api.py` | `/chat`, input validation, SSE stream, history and delete, follow-ups using memory, upload ingest and rejection, health ok / degraded, in-band stream errors |

---

## 13. Logging, errors and security

**Structured logs.** With `LOG_JSON=true` every log line is a JSON object, for example:

```json
{"ts": "...", "level": "INFO", "logger": "app.main", "message": "request",
 "request_id": "3f9c1a2b4d5e", "method": "POST", "path": "/chat", "status": 200, "duration_ms": 1432.7}
```

Every response has an **`x-request-id`** header (or echoes the one you send),
and the same id appears in the logs and in error bodies, so you can trace a
failed request.

**Consistent error responses.** Every error has the shape
`{"error": "...", "detail": "...", "request_id": "..."}`:

| Situation | Status | `error` |
|---|---|---|
| Invalid input | 422 | `validation_error` |
| Wrong / revoked OpenAI key | 502 | `openai_authentication_failed` |
| OpenAI rate limit or no credit | 429 | `openai_rate_limited` |
| Other OpenAI failure | 502 | `openai_error` |
| Missing `OPENAI_API_KEY` | 503 | `service_unavailable` |
| Anything unexpected | 500 | `internal_error` (no stack trace leaked to the client) |

**Security measures**

- The API key is a `SecretStr` and never printed; logs also **redact** anything that looks like an OpenAI key (`sk-...`).
- `.env` is git-ignored and Docker-ignored; only `.env.example` (no secrets) is shared.
- Uploads are type- and size-checked, written to a temporary folder that gets deleted, and their names are sanitized against path traversal.
- Session ids are validated against a strict pattern.
- Containers run as a non-root user (`appuser`).
- CORS is limited to the UI's origin by default.
- Prompt rules tell the model to use only document context, which limits hallucination. Note that a document *containing* instructions could still influence the model (prompt injection), so only index documents you trust.

> The API has **no authentication**. It is meant for local or internal use;
> put it behind an authenticating proxy before exposing it to the internet.

---

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/health` shows `openai_api_key: ok=false`, chat returns 503 | `OPENAI_API_KEY` not set. Add it to `.env` and restart (`docker compose up -d`). |
| Chat returns `openai_authentication_failed` | The key is wrong or revoked. Create a new one. |
| Chat returns `openai_rate_limited` | Rate limit hit or account out of credit. Check billing at platform.openai.com. |
| Every answer is *"I don't know based on the provided documents."* | Nothing is indexed yet (run ingestion), or the question really isn't covered. Also check `SCORE_THRESHOLD` isn't set too high. |
| `Bind for ...:5432 failed: port is already allocated` (or 8000 / 6379 / 8501) | Another program or container is already using that port. Stop it, or change the **left** side of the port mapping in `docker-compose.yml` (e.g. `"8001:8000"`) and open that port instead. Postgres doesn't need a host port at all when everything runs in Compose. |
| `/health` shows `postgres: ok=false` | Database not running or wrong `DATABASE_URL`. Use host `localhost` when running locally and `postgres` inside Compose. |
| Redis errors with `MEMORY_BACKEND=redis` | Redis needs the JSON + Search modules: use Redis 8+ or Redis Stack (the Compose file uses `redis:8`). |
| UI says *Could not reach the API* | API not running, or `API_URL` is wrong for where the UI runs. |
| Upload rejected with 400 / 413 | Only `.pdf .docx .txt .md` are accepted; raise `MAX_UPLOAD_MB` for big files. |
| Answers got worse after changing the embedding model or chunk size | Old vectors are incompatible. Re-ingest into a new `COLLECTION_NAME`, or `make clean` and re-ingest. |
| **Windows:** `DLL load failed while importing _xxhash: An Application Control policy has blocked this file` | **Windows Smart App Control** is blocking a native library that LangChain depends on (`langsmith` → `xxhash`). Run the project and tests in Docker (`make up`, `make test-docker`) or inside WSL. |
| Scanned PDF gives no answers | Image-only pages have no text to extract and are skipped. Run OCR on the PDF first. |

---

## 15. Deprecation notes

These warnings are known and filtered in `pyproject.toml`:

- **`RunnableWithMessageHistory`** is deprecated in langchain-core 1.x (to be removed in 2.0). This project doesn't use it: `ConversationalRAG` loads, trims and saves history explicitly, which is clearer and works with any history backend. The chat-history classes themselves still emit deprecation warnings, which are expected.
- **`langchain-community`** is being sunset. It is used only for the document loaders (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`) in `app/ingestion/loaders.py`, which is the single place to change when replacements are available.
- `langchain-redis` 0.2.x doesn't support Python 3.14 yet, hence `requires-python = ">=3.11,<3.14"`.

---

## 16. Ideas for extending the project

- **Authentication**: API keys or OAuth in front of the endpoints, with sessions tied to users.
- **Document management**: list indexed files and delete a file's chunks (each chunk already stores `source`).
- **Hybrid search**: combine vector similarity with keyword search (Postgres full-text) for exact terms like product codes.
- **Re-ranking**: fetch more chunks and re-rank them with a cross-encoder before answering.
- **Conversation summaries**: summarize old turns instead of dropping them when trimming history.
- **Evaluation**: a small question/expected-answer set to measure answer quality when you change prompts or settings.
- **OCR** for scanned PDFs, and more formats (HTML, PPTX, CSV).
- **Connection pooling** (`psycopg_pool`) for many concurrent users.
