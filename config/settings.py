"""Application settings loaded from environment variables."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "document_search"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Phoenix
    phoenix_host: str = "localhost"
    phoenix_port: int = 6006

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Prompts
    prompts_dir: str = "./prompts"

    class Config:
        env_file = ".env"

settings = Settings()
