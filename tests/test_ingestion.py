import pytest
from unittest.mock import patch, MagicMock
from ingestion.docling_processor import DoclingProcessor
from ingestion.embedder import DocumentEmbedder

def test_docling_processor_file_not_found():
    processor = DoclingProcessor()
    with pytest.raises(FileNotFoundError):
        processor.process_pdf("/nonexistent/file.pdf")

def test_docling_processor_empty_directory(tmp_path):
    processor = DoclingProcessor()
    result = processor.process_directory(str(tmp_path))
    assert result == []

def test_embedder_chunk_documents():
    embedder = DocumentEmbedder()
    raw_docs = [
        {"text": "This is a test document.", "metadata": {"filename": "test.pdf"}, "filename": "test.pdf"}
    ]
    docs = embedder.chunk_documents(raw_docs)
    assert len(docs) == 1
    assert docs[0].text == "This is a test document."
