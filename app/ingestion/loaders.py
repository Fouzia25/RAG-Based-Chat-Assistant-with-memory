"""Document loaders for PDF, TXT, MD and DOCX files.

Every returned ``Document`` carries normalized metadata:
``source`` (display filename), ``page`` (1-based) and ``file_type``.
Formats without real pages (TXT/MD/DOCX) are reported as page 1.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class UnsupportedFileTypeError(ValueError):
    pass


def _get_loader(path: Path) -> BaseLoader:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path), mode="page")
    if ext in {".txt", ".md"}:
        # Markdown is kept as plain text: headings and lists remain readable
        # for the LLM, and we avoid the heavy `unstructured` dependency.
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def load_file(path: str | Path, display_name: str | None = None) -> list[Document]:
    """Load a single file.

    ``display_name`` overrides the filename stored as ``source`` (used by the
    API, where uploads are written to a temp path first).
    """
    path = Path(path)
    source = display_name or path.name
    docs = _get_loader(path).load()

    normalized: list[Document] = []
    for i, doc in enumerate(docs):
        if not doc.page_content.strip():
            continue  # e.g. blank or image-only PDF pages
        # PyPDFLoader's "page" is 0-based; humans count from 1.
        raw_page = doc.metadata.get("page")
        page = int(raw_page) + 1 if isinstance(raw_page, int) else i + 1
        normalized.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": source, "page": page, "file_type": path.suffix.lower().lstrip(".")},
            )
        )
    return normalized


def discover_files(root: str | Path) -> list[Path]:
    """Return supported files under ``root`` (a file or a directory, recursive)."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
