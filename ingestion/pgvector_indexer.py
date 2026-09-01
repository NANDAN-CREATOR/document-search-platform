"""Index nodes into ChromaDB using LlamaIndex."""
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings

from config.settings import settings

logger = logging.getLogger(__name__)


class ChromaIndexer:
    """Index embeddings into ChromaDB using LlamaIndex."""

    def __init__(self):
        # Configure LlamaIndex global settings
        LlamaSettings.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
        )
        LlamaSettings.embed_model = OllamaEmbedding(
            model_name=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

        # Create persist directory
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )

        # Get or create collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name
        )

        # Build LlamaIndex ChromaVectorStore
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        logger.info(
            f"ChromaDB initialized at: {settings.chroma_persist_dir} "
            f"| Collection: {settings.chroma_collection_name}"
        )

    def index_nodes(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Index nodes into ChromaDB."""
        logger.info(f"Indexing {len(nodes)} nodes into ChromaDB...")
        index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=True,
        )
        logger.info("Indexing complete.")
        return index

    def load_index(self) -> VectorStoreIndex:
        """Load existing index from ChromaDB."""
        logger.info("Loading index from ChromaDB...")
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
        )
        return index

    def get_retriever(self, similarity_top_k: Optional[int] = None):
        """Get a retriever from the loaded index."""
        top_k = similarity_top_k or settings.similarity_top_k
        index = self.load_index()
        return index.as_retriever(similarity_top_k=top_k)

    def get_doc_count(self) -> int:
        """Get number of documents in collection."""
        return self.chroma_collection.count()
