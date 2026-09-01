# Document Search Platform

Agentic RAG Document Search Platform with OpenWebUI integration.

## Tech Stack

| Component | Tool |
|-----------|------|
| Document Preprocessing | Docling |
| Vector Database | PostgreSQL + PGVector |
| RAG Framework | LlamaIndex |
| Multi-Agent | CrewAI |
| LLM Provider | Ollama |
| Tracing & Observability | Arize Phoenix |
| RAG Evaluation | RAGAs |
| Frontend | OpenWebUI |
| API | FastAPI |

## Project Structure

```
document-search-platform/
├── ingestion/         # Document ingestion pipeline
├── agents/            # CrewAI multi-agent setup
├── api/               # FastAPI REST endpoints
├── prompts/           # Externalized prompt templates
├── evaluation/        # RAGAs evaluation pipeline
├── tracing/           # Arize Phoenix tracing setup
├── config/            # Configuration files
├── doc/               # Architecture diagrams, presentation, API docs
├── tests/             # Unit and integration tests
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Setup & Installation

> Detailed setup instructions coming soon. See `/doc` for architecture and design documents.

## API Documentation

> Swagger/OpenAPI documentation available at `/doc/api_swagger.yaml`
