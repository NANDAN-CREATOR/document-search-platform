"""Main ingestion pipeline orchestrator."""
import logging
from pathlib import Path

from ingestion.docling_processor import DoclingProcessor
from ingestion.embedder import DocumentEmbedder
from ingestion.pgvector_indexer import PGVectorIndexer
from config.settings import settings
from tracing.phoenix_setup import instrument_llamaindex

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full document ingestion workflow."""

    def __init__(self):
        self.processor = DoclingProcessor()
        self.embedder = DocumentEmbedder()
        self.indexer = PGVectorIndexer()

    def run(self, data_dir: str = None) -> dict:
        """Run the full ingestion pipeline.

        Pipeline:
        PDF files -> Docling preprocessing ->
        LlamaIndex chunking + Ollama embedding ->
        PGVector indexing
        """
        data_dir = data_dir or settings.data_dir

        logger.info(f"Starting ingestion pipeline for: {data_dir}")

        # Step 1: Preprocess PDFs with Docling
        logger.info("Step 1: Preprocessing PDFs with Docling...")
        raw_docs = self.processor.process_directory(data_dir)
        if not raw_docs:
            return {"status": "error", "message": "No documents found"}

        # Step 2: Chunk and embed with LlamaIndex + Ollama
        logger.info("Step 2: Chunking and embedding documents...")
        nodes = self.embedder.process(raw_docs)

        # Step 3: Index into PGVector
        logger.info("Step 3: Indexing into PGVector...")
        self.indexer.index_nodes(nodes)

        result = {
            "status": "success",
            "documents_processed": len(raw_docs),
            "chunks_indexed": len(nodes),
            "data_dir": str(data_dir),
        }
        logger.info(f"Ingestion complete: {result}")
        return result


def run_ingestion(data_dir: str = None) -> dict:
    """Convenience function to run ingestion pipeline."""
    # Instrument LlamaIndex with Phoenix tracing
    instrument_llamaindex()

    pipeline = IngestionPipeline()
    return pipeline.run(data_dir)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else settings.data_dir
    result = run_ingestion(data_directory)
    print(result)
