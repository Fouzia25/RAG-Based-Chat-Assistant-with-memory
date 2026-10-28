"""Ingestion pipeline: load -> split -> enrich metadata -> dedupe -> embed + upsert.

CLI::

    python -m app.ingestion.ingest --path ./data
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.loaders import discover_files, load_file
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class FileIngestionResult:
    filename: str
    chunks_total: int = 0
    chunks_added: int = 0
    chunks_skipped: int = 0
    error: str | None = None


@dataclass
class IngestionResult:
    files: list[FileIngestionResult] = field(default_factory=list)

    @property
    def chunks_added(self) -> int:
        return sum(f.chunks_added for f in self.files)

    @property
    def chunks_skipped(self) -> int:
        return sum(f.chunks_skipped for f in self.files)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(chunk: Document) -> str:
    """Deterministic id: re-ingesting the same file produces the same ids (upsert)."""
    meta = chunk.metadata
    return content_hash(f"{meta['source']}|{meta['page']}|{meta['content_hash']}")


def split_documents(
    docs: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Document]:
    """Split pages into chunks and add chunk_index, content_hash and upload_timestamp."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    next_index: dict[str, int] = defaultdict(int)  # chunk index is per source file
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk_index"] = next_index[source]
        next_index[source] += 1
        chunk.metadata["content_hash"] = content_hash(chunk.page_content)
        chunk.metadata["upload_timestamp"] = uploaded_at
    return chunks


def _existing_ids(vectorstore: VectorStore, ids: list[str]) -> set[str]:
    try:
        return {doc.id for doc in vectorstore.get_by_ids(ids) if doc.id}
    except NotImplementedError:
        return set()


def index_chunks(chunks: list[Document], vectorstore: VectorStore) -> tuple[int, int]:
    """Embed and upsert chunks not already stored. Returns (added, skipped)."""
    unique: dict[str, Document] = {}
    for chunk in chunks:
        unique.setdefault(chunk_id(chunk), chunk)  # drops duplicates within the batch

    ids = list(unique)
    already_stored = _existing_ids(vectorstore, ids) if ids else set()
    new_ids = [i for i in ids if i not in already_stored]

    if new_ids:
        # Only new chunks are embedded, so re-uploading a file costs nothing.
        vectorstore.add_documents([unique[i] for i in new_ids], ids=new_ids)
    return len(new_ids), len(chunks) - len(new_ids)


def ingest_files(
    files: list[tuple[Path, str]], vectorstore: VectorStore | None = None
) -> IngestionResult:
    """Ingest ``(path_on_disk, display_name)`` pairs into the vector store."""
    if vectorstore is None:
        from app.rag.retriever import get_vectorstore

        vectorstore = get_vectorstore()

    result = IngestionResult()
    for path, display_name in files:
        file_result = FileIngestionResult(filename=display_name)
        try:
            chunks = split_documents(load_file(path, display_name=display_name))
            file_result.chunks_total = len(chunks)
            file_result.chunks_added, file_result.chunks_skipped = index_chunks(chunks, vectorstore)
            logger.info(
                "file_ingested",
                extra={
                    "file": display_name,
                    "chunks_added": file_result.chunks_added,
                    "chunks_skipped": file_result.chunks_skipped,
                },
            )
        except Exception as exc:  # one bad file should not abort the batch
            file_result.error = str(exc)
            logger.exception("file_ingestion_failed", extra={"file": display_name})
        result.files.append(file_result)
    return result


def ingest_path(path: str | Path, vectorstore: VectorStore | None = None) -> IngestionResult:
    files = discover_files(path)
    return ingest_files([(p, p.name) for p in files], vectorstore=vectorstore)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index documents into the PGVector store.")
    parser.add_argument("--path", required=True, help="File or directory to ingest (recursive).")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, json_format=False)

    if not Path(args.path).exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 1
    if not discover_files(args.path):
        print(f"No supported files (.pdf, .txt, .md, .docx) found under {args.path}", file=sys.stderr)
        return 1

    result = ingest_path(args.path)
    for f in result.files:
        status = f"ERROR: {f.error}" if f.error else f"{f.chunks_added} added, {f.chunks_skipped} skipped"
        print(f"  {f.filename}: {status}")
    print(
        f"Done. {result.chunks_added} chunks added, {result.chunks_skipped} already indexed "
        f"(collection '{settings.collection_name}')."
    )
    return 1 if any(f.error for f in result.files) else 0


if __name__ == "__main__":
    sys.exit(main())
