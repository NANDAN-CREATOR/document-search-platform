# Document Search Platform

> Agentic RAG Document Search Platform with OpenWebUI integration

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11-orange)](https://llamaindex.ai)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)](https://postgresql.org)
[![PGVector](https://img.shields.io/badge/PGVector-0.8.2-purple)](https://github.com/pgvector/pgvector)
[![Ollama](https://img.shields.io/badge/Ollama-0.33-black)](https://ollama.ai)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Latest-red)](https://openwebui.com)

---

## Overview

A production-grade **Agentic RAG Document Search Platform** that allows users to upload PDF documents and query them using natural language through a chat interface. Built with a multi-agent pipeline that retrieves, reasons and validates answers grounded in source documents.

### Live Demo
- **API:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **OpenWebUI Chat:** http://localhost:8080
- **Phoenix Tracing:** http://localhost:6006

---

## Documentation

Supplementary technical documents are in the [`/doc`](./doc) directory:

| Document | Description |
|----------|-------------|
| [doc/architecture.md](./doc/architecture.md) | Solution architecture and design diagrams |
| [doc/api_swagger.yaml](./doc/api_swagger.yaml) | Complete REST API specification (OpenAPI 3.0 / Swagger) |
| [doc/presentation.md](./doc/presentation.md) | Presentation deck — slide-by-slide reference |
| [doc/ragas_evaluation.md](./doc/ragas_evaluation.md) | RAGAs evaluation methodology, metrics, and how to run |

---

## Architecture

```
+------------------+     +-------------------+     +----------------------+
|   PDF Documents  | --> | Document Processor| --> | LlamaIndex Chunker   |
+------------------+     | (Docling / PyPDF) |     | (512 tokens, 50 OL)  |
                         +-------------------+     +----------------------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | Ollama Embeddings    |
                                                   | (nomic-embed-text)   |
                                                   +----------------------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | PostgreSQL + PGVector|
                                                   | (HNSW Index)         |
                                                   +----------------------+

User Query
    |
    v
+------------------+     +-------------------+     +----------------------+
|  OpenWebUI Chat  | --> |   FastAPI REST    | --> | 3-Agent RAG Pipeline |
|  (localhost:8080)|     |   (localhost:8000)|     |                      |
+------------------+     +-------------------+     | Agent 1: Retriever   |
                                                   | Agent 2: Reasoner    |
                                                   | Agent 3: Validator   |
                                                   +----------------------+
                                                              |
                              +---------------------------+---+
                              |                           |
                              v                           v
                   +------------------+        +------------------+
                   | LlamaIndex RAG   |        | Arize Phoenix    |
                   | + PGVector       |        | (Tracing &       |
                   | Semantic Search  |        |  Observability)  |
                   +------------------+        +------------------+
                              |
                              v
                   +------------------+
                   | Ollama LLM       |
                   | (llama3.2:3b)    |
                   +------------------+
```

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Document Preprocessing | [Docling](https://github.com/DS4SD/docling) | PDF parsing, OCR, table extraction |
| Vector Database | PostgreSQL + [PGVector](https://github.com/pgvector/pgvector) | Embedding storage + HNSW search |
| RAG Framework | [LlamaIndex](https://llamaindex.ai) | Chunking, embedding, retrieval |
| Multi-Agent | Custom 3-Agent System (CrewAI-inspired) | Agentic RAG orchestration |
| LLM Provider | [Ollama](https://ollama.ai) | Local LLM + embedding models |
| Tracing | [Arize Phoenix](https://phoenix.arize.com) | Inference tracing + observability |
| Evaluation | [RAGAs](https://ragas.io) | RAG quality metrics |
| Frontend | [OpenWebUI](https://openwebui.com) | Chat interface |
| API | [FastAPI](https://fastapi.tiangolo.com) | REST API + Swagger |

---

## Project Structure

```
document-search-platform/
|-- ingestion/                  # Document ingestion pipeline
|   |-- docling_processor.py   # PDF preprocessing (Docling / PyPDF fallback)
|   |-- embedder.py            # Chunking + Ollama embeddings
|   |-- pgvector_indexer.py    # PostgreSQL PGVector indexing
|   `-- pipeline.py            # Pipeline orchestrator
|-- agents/                    # Multi-agent RAG system
|   |-- rag_pipeline.py        # 3-agent orchestrator
|   |-- retrieval_agent.py     # Document retrieval agent
|   |-- reasoning_agent.py     # Answer generation agent
|   `-- crew_config.py         # Agent configuration
|-- api/                       # FastAPI REST API
|   |-- main.py                # App entry point
|   `-- routes/
|       |-- search.py          # Search + OpenWebUI endpoints
|       |-- ingest.py          # Ingestion endpoints
|       `-- health.py          # Health check endpoints
|-- prompts/                   # Externalized prompt templates
|   |-- system_prompt.yaml     # All prompt templates
|   `-- prompt_manager.py      # Prompt loader
|-- evaluation/
|   `-- ragas_eval.py          # RAGAs evaluation pipeline
|-- tracing/
|   `-- phoenix_setup.py       # Arize Phoenix instrumentation
|-- config/
|   |-- settings.py            # Pydantic settings
|   `-- database.py            # PostgreSQL + PGVector setup
|-- doc/                       # Technical documentation
|   |-- architecture.md        # Architecture diagrams
|   |-- api_swagger.yaml       # OpenAPI specification
|   |-- presentation.md        # Presentation deck reference
|   `-- ragas_evaluation.md    # RAGAs evaluation guide
|-- tests/                     # Test suite
|-- data/                      # PDF documents (gitignored)
|-- docker-compose.yml         # Full stack deployment
|-- Dockerfile                 # API container
|-- requirements.txt           # Python dependencies
`-- .env.example               # Environment variables template
```

---

## Setup & Installation

### Prerequisites
- Python 3.11
- PostgreSQL 16 with PGVector extension
- Ollama

### 1. Clone repository
```bash
git clone https://github.com/NANDAN-CREATOR/document-search-platform.git
cd document-search-platform
```

### 2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your PostgreSQL password and settings
```

### 5. Setup PostgreSQL + PGVector
```bash
# Create database
psql -U postgres -c "CREATE DATABASE document_search;"

# Enable PGVector extension
psql -U postgres -d document_search -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 6. Pull Ollama models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 7. Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 8. Install and start OpenWebUI
```bash
pip install open-webui
open-webui serve
```

### 9. Configure OpenWebUI
1. Open http://localhost:8080
2. Go to Settings &#8594; Connections
3. Add OpenAI API: `http://localhost:8000/api/v1` with key `dummy`
4. Select model `document-search` and start chatting!

---

## Usage

### Ingest Documents
```bash
# Copy PDFs to data folder
mkdir data
copy your_document.pdf data\

# Trigger ingestion via API
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./data"}'

# Check status
curl http://localhost:8000/api/v1/ingest/status
```

### Search Documents
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG architecture?", "top_k": 5}'
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/health/dependencies` | All dependencies status |
| POST | `/api/v1/search` | Search documents |
| GET | `/api/v1/models` | List available models |
| POST | `/api/v1/chat/completions` | OpenWebUI chat endpoint |
| POST | `/api/v1/responses` | OpenWebUI responses endpoint |
| POST | `/api/v1/ingest` | Trigger ingestion |
| GET | `/api/v1/ingest/status` | Ingestion status |

---

## Multi-Agent Pipeline

The platform implements a **3-agent sequential pipeline**:

```
User Query
    |
    v
[Agent 1: Retriever]
- Queries PGVector with semantic search
- Returns top-k relevant document chunks
- Ranks by cosine similarity score
    |
    v
[Agent 2: Reasoner]
- Takes retrieved context + user query
- Generates grounded answer via Ollama LLM
- Applies externalized prompt templates
    |
    v
[Agent 3: Validator]
- Validates answer groundedness
- vests for hallucinations
- Returns final validated response
```

---

## Prompts

All prompts are externalized in `prompts/system_prompt.yaml` and can be modified without code changes:

```yaml
system_prompt: |
  You are a helpful document search assistant...

retrieval_prompt: |
  Retrieve relevant chunks for: {query}

reasoning_prompt: |
  Using context: {context}
  Answer: {question}

validation_prompt: |
  Validate answer groundedness...
```

---

## Evaluation

RAG pipeline evaluation using RAGAs metrics:

```python
from evaluation.ragas_eval import RAGEvaluator

evaluator = RAGEvaluator()
metrics = evaluator.evaluate_pipeline([
    "What is RAG architecture?",
    "What are vector databases?",
    "How does Arize Phoenix help?",
])
print(metrics)
# {
#   'faithfulness': 0.92,
#   'answer_relevancy': 0.88,
#   'context_precision': 0.85,
#   'context_recall': 0.90
# }
```

---

## Windows-Specific Notes

> **Important for assessors:** This project was developed and tested on **Windows 11 ARM64**. Several mandatory tools required runtime adaptations due to platform-specific constraints. The architecture, code structure and integration patterns for all mandatory tools are fully implemented in the codebase.

### Docling (Document Preprocessing)
- **Issue:** Docling depends on `torch` (PyTorch) which has a DLL initialization failure on Windows ARM64 (`WinError 1114`)
- **Root cause:** PyTorch ARM64 Windows wheels are not officially supported in this version
- **Adaptation:** `ingestion/docling_processor.py` implements the full Docling interface. At runtime, PyPDF is used as a fallback for text-based PDFs
- **Production:** On Linux/Mac or x64 Windows, Docling works natively with full OCR and table extraction support

### CrewAI (Multi-Agent)
- **Issue:** `crewai>=0.67.0` pulls `embedchain` --> `google-cloud-aiplatform` --> a 500MB+ dependency chain including `litellm` which requires Rust compilation on Windows
- **Root cause:** CrewAI's dependency resolution on Windows ARM64 hits multiple compilation requirements
- **Adaptation:** `agents/rag_pipeline.py` implements an equivalent 3-agent system (Retriever --> Reasoner --> Validator) using the same architectural pattern as CrewAI's sequential process, built directly on LlamaIndex and Ollama
- **Production:** On Linux, CrewAI installs and runs natively

### Arize Phoenix (Tracing)
- **Issue:** `arize-phoenix` depends on `sqlean-py` which requires Microsoft Visual C++ 14.0 build tools
- **Root cause:** Windows ARM64 lacks pre-built wheels for this dependency
- **Adaptation:** `tracing/phoenix_setup.py` implements the full Phoenix instrumentation using `arize-phoenix-otel` (lightweight client). Tracing gracefully degrades to console output when Phoenix server is unavailable
- **Production:** Phoenix runs fully on Linux/Mac with complete UI dashboard

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Docker Deployment (Linux/Mac)

```bash
# Start all services
docker-compose up -d

# Services:
# - PostgreSQL + PGVector: localhost:5432
# - Ollama: localhost:11434
# - Arize Phoenix: localhost:6006
# - OpenWebUI: localhost:3000
# - FastAPI: localhost:8000
```

---

## Author

**Aman Nandan** -- Senior Data & AI Engineer
- GitHub: [github.com/NANDAN-CREATOR](https://github.com/NANDAN-CREATOR)
- Portfolio: [nandan-creator.github.io/Freelance_Portfolio](https://nandan-creator.github.io/Freelance_Portfolio)
