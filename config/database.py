"""Database connection and PGVector setup."""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.settings import settings

logger = logging.getLogger(__name__)


def get_engine():
    """Create SQLAlchemy engine."""
    return create_engine(
        settings.postgres_connection_string,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def get_session():
    """Create SQLAlchemy session."""
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_pgvector(engine=None):
    """Initialize PGVector extension and tables."""
    if engine is None:
        engine = get_engine()

    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("PGVector extension enabled.")

        # Create embeddings table
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {settings.vector_table_name} (
                id SERIAL PRIMARY KEY,
                node_id VARCHAR(255) UNIQUE NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                embedding vector({settings.embedding_dimension}),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
        logger.info(f"Table '{settings.vector_table_name}' ready.")

        # Create HNSW index for fast similarity search
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
            ON {settings.vector_table_name}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        conn.commit()
        logger.info("HNSW index created.")

    return engine


def check_db_health() -> bool:
    """Check database connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
