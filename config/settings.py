"""Application settings loaded from environment variables."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # ChromaDB (zero setup - stores locally)
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "document_embeddings"

    # Arize Phoenix
    phoenix_host: str = "localhost"
    phoenix_port: int = 6006
    phoenix_grpc_port: int = 4317

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Prompts
    prompts_dir: str = "./prompts"

    # Data
    data_dir: str = "./data"

    # Vector Store
    similarity_top_k: int = 5

    @property
    def phoenix_endpoint(self) -> str:
        return f"http://{self.phoenix_host}:{self.phoenix_grpc_port}"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
