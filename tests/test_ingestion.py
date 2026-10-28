from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingestion.ingest import chunk_id, index_chunks, ingest_files, ingest_path, split_documents
from app.ingestion.loaders import UnsupportedFileTypeError, discover_files, load_file

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


class FakeVectorStore:
    """Stands in for PGVector: stores docs by id, no embeddings."""

    def __init__(self):
        self.docs: dict[str, Document] = {}
        self.add_calls = 0

    def add_documents(self, documents, ids):
        self.add_calls += 1
        for doc, id_ in zip(documents, ids, strict=True):
            self.docs[id_] = doc
        return ids

    def get_by_ids(self, ids):
        return [Document(id=i, page_content=self.docs[i].page_content) for i in ids if i in self.docs]


@pytest.fixture
def text_file(tmp_path) -> Path:
    path = tmp_path / "notes.txt"
    # Distinct sentences so every chunk is unique (identical chunks are deduplicated).
    paragraphs = [" ".join(f"Policy {p}.{s} covers topic {p * 100 + s}." for s in range(30)) for p in range(5)]
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


def test_load_txt_sets_normalized_metadata(text_file):
    docs = load_file(text_file)
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "notes.txt", "page": 1, "file_type": "txt"}


def test_load_file_uses_display_name(text_file):
    docs = load_file(text_file, display_name="original-upload.txt")
    assert docs[0].metadata["source"] == "original-upload.txt"


def test_load_markdown_sample():
    docs = load_file(SAMPLE_DIR / "company_handbook.md")
    assert docs and docs[0].metadata["file_type"] == "md"


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "image.png"
    bad.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileTypeError):
        load_file(bad)


def test_discover_files_filters_extensions(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.TXT").write_text("y")
    (tmp_path / "c.csv").write_text("z")
    assert sorted(p.name for p in discover_files(tmp_path)) == ["a.md", "b.TXT"]


def test_split_adds_chunk_metadata(text_file):
    chunks = split_documents(load_file(text_file), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert len(c.page_content) <= 300
        assert len(c.metadata["content_hash"]) == 64
        assert c.metadata["upload_timestamp"]
        assert c.metadata["source"] == "notes.txt"
        assert c.metadata["page"] == 1


def test_chunk_ids_are_deterministic(text_file):
    a = split_documents(load_file(text_file), chunk_size=300, chunk_overlap=50)
    b = split_documents(load_file(text_file), chunk_size=300, chunk_overlap=50)
    assert [chunk_id(c) for c in a] == [chunk_id(c) for c in b]


def test_reingesting_same_file_is_deduplicated(text_file):
    store = FakeVectorStore()

    first = ingest_files([(text_file, "notes.txt")], vectorstore=store)
    second = ingest_files([(text_file, "notes.txt")], vectorstore=store)

    assert first.chunks_added > 0 and first.chunks_skipped == 0
    assert second.chunks_added == 0
    assert second.chunks_skipped == first.chunks_added
    assert store.add_calls == 1  # nothing re-embedded the second time


def test_duplicate_chunks_within_batch_are_collapsed():
    meta = {"source": "a.txt", "page": 1, "content_hash": "h"}
    chunks = [Document(page_content="same", metadata=dict(meta)) for _ in range(3)]
    store = FakeVectorStore()
    added, skipped = index_chunks(chunks, store)
    assert (added, skipped) == (1, 2)


def test_bad_file_does_not_abort_batch(tmp_path, text_file):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    result = ingest_files([(bad, "broken.pdf"), (text_file, "notes.txt")], vectorstore=FakeVectorStore())
    assert result.files[0].error
    assert result.files[1].error is None and result.files[1].chunks_added > 0


def test_ingest_sample_directory():
    result = ingest_path(SAMPLE_DIR, vectorstore=FakeVectorStore())
    assert {f.filename for f in result.files} >= {"company_handbook.md", "product_faq.txt"}
    assert result.chunks_added > 0
