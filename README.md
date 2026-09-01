# Document Search Platform

> Agentic RAG Document Search Platform with OpenWebUI integration

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11-orange)](https://llamaindex.ai)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.67-red)](https://crewai.com)

## Overview

A production-grade document search platform that combines:
- **Docling** for intelligent PDF preprocessing
- **LlamaIndex + PGVector** for semantic retrieval
- **CrewAI multi-agents** for agentic RAG (Retriever → Reasoner → Validator)
- **Arize Phoenix** for full inference tracing and observability
- **RAGAs** for automated pipeline evaluation
- **FastAPI** REST API compatible with **OpenWebUI**

## Architecture

See [`/doc/architecture.md`](doc/architecture.md) for detailed diagrams.

```
PDF Docs → Docling → LlamaIndex Chunks → Ollama Embeddings → PGVector
                                                                   ↓
OpenWebUI ← FastAPI ← CrewAI Agents (Retriever→Reasoner→Validator)
                              ↓
                      Arize Phoenix (Tracing)
```

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Document Preprocessing | [Docling](https://github.com/DS4SD/docling) | PDF parsing, OCR, table extraction |
| Vector Database | PostgreSQL + [PGVector](https://github.com/pgvector/pgvector) | Embedding storage + HNSW search |
| RAG Framework | [LlamaIndex](https://llamaindex.ai) | Chunking, embedding, retrieval |
| Multi-Agent | [CrewAI](https://crewai.com) | Agentic RAG orchestration |
| LLM Provider | [Ollama](https://ollama.ai) | Local LLM + embedding models |
| Tracing | [Arize Phoenix](https://phoenix.arize.com) | Inference tracing + PromptOps |
| Evaluation | [RAGAs](https://ragas.io) | RAG quality metrics |
| Frontend | [OpenWebUI](https://openwebui.com) | Chat interface |
| API | [FastAPI](https://fastapi.tiangolo.com) | REST API + Swagger |

## Project Structure

```
document-search-platform/
├── ingestion/                 # Document ingestion pipeline
│   ├── docling_processor.py   # PDF preprocessing with Docling
│   ├── embedder.py            # Chunking + Ollama embeddings
│   ├── pgvector_indexer.py    # PGVector indexing
│   └── pipeline.py            # Pipeline orchestrator
├── agents/                    # CrewAI multi-agent system
│   ├── crew_config.py         # Agent definitions
│   ├── retrieval_agent.py     # Document retrieval agent
│   ├── reasoning_agent.py     # Answer generation agent
│   └── rag_pipeline.py        # Agentic RAG orchestrator
├── api/                       # FastAPI REST API
│   ├── main.py                # App entry point
│   └── routes/
│       ├── search.py          # Search + OpenWebUI endpoints
│       ├── ingest.py          # Ingestion endpoints
│       └── health.py          # Health check endpoints
├── prompts/                   # Externalized prompt templates
│   ├── system_prompt.yaml     # All prompt templates
│   └── prompt_manager.py      # Prompt loader
├── evaluation/
│   └── ragas_eval.py          # RAGAs evaluation pipeline
├── tracing/
│   └── phoenix_setup.py       # Arize Phoenix instrumentation
├── config/
│   ├── settings.py            # Pydantic settings
│   └── database.py            # PostgreSQL + PGVector setup
├── doc/                       # Technical documentation
│   ├── architecture.md        # Architecture diagrams
│   └── api_swagger.yaml       # OpenAPI specification
├── tests/                     # Test suite
├── docker-compose.yml         # Full stack deployment
├── Dockerfile                 # API container
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variables template
```

## Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Ollama installed locally

### 1. Clone and configure

```bash
git clone https://github.com/NANDAN-CREATOR/document-search-platform.git
cd document-search-platform
cp .env.example .env
# Edit .env with your configuration
```

### 2. Pull Ollama models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 3. Start with Docker Compose

```bash
docker-compose up -d
```

This starts:
- PostgreSQL + PGVector on port `5432`
- Ollama on port `11434`
- Arize Phoenix on port `6006`
- OpenWebUI on port `3000`
- FastAPI on port `8000`

### 4. Add your PDF documents

```bash
mkdir -p data/
cp /path/to/your/documents/*.pdf data/
```

### 5. Run document ingestion

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"data_dir": "./data"}'

# Or directly
python -m ingestion.pipeline
```

### 6. Search documents

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the main topic?"}'
```

### 7. Open OpenWebUI

Navigate to `http://localhost:3000` and start chatting with your documents!

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: [`/doc/api_swagger.yaml`](doc/api_swagger.yaml)

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health |
| `GET` | `/health/dependencies` | All dependencies health |
| `POST` | `/api/v1/search` | Search documents |
| `POST` | `/api/v1/v1/chat/completions` | OpenWebUI chat endpoint |
| `POST` | `/api/v1/ingest` | Trigger ingestion |
| `GET` | `/api/v1/ingest/status` | Ingestion status |

## Observability

Arize Phoenix dashboard: http://localhost:6006

Traces all:
- LlamaIndex retrieval calls
- CrewAI agent steps
- LLM inference calls
- Embedding generation

## Evaluation

```python
from evaluation.ragas_eval import RAGEvaluator

evaluator = RAGEvaluator()
metrics = evaluator.evaluate_pipeline(
    questions=[
        "What is the main topic?",
        "What are the key findings?",
    ]
)
print(metrics)
# {
#   'faithfulness': 0.92,
#   'answer_relevancy': 0.88,
#   'context_precision': 0.85,
#   'context_recall': 0.90
# }
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Configuration

All configuration via `.env` file — see `.env.example` for all options.
Prompt templates in `prompts/system_prompt.yaml` — edit without code changes.

## Author

**Aman Nandan** — Senior Data & AI Engineer
- GitHub: [github.com/NANDAN-CREATOR](https://github.com/NANDAN-CREATOR)
- Portfolio: [nandan-creator.github.io/Freelance_Portfolio](https://nandan-creator.github.io/Freelance_Portfolio)
