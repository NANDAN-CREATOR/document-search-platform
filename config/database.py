"""ChromaDB health check utility."""
import logging

logger = logging.getLogger(__name__)


def check_db_health() -> bool:
    """Check ChromaDB health by verifying it can be imported and initialized."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        client.heartbeat()
        return True
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        return False
