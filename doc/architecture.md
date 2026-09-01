# Solution Architecture

## System Overview

The Document Search Platform is an Agentic RAG system built on a modern AI stack.

## Architecture Diagram

```
+------------------+     +-------------------+     +------------------+
|   PDF Documents  | --> | Docling Processor | --> | Document Chunks  |
+------------------+     +-------------------+     +------------------+
                                                           |
                                                           v
                                                  +------------------+
                                                  | Ollama Embedding |
                                                  | (nomic-embed)    |
                                                  +------------------+
                                                           |
                                                           v
                                                  +------------------+
                                                  | PGVector Index   |
                                                  | (PostgreSQL)     |
                                                  +------------------+


User Query
    |
    v
+------------------+     +-------------------+     +------------------+
|  OpenWebUI Chat  | --> |   FastAPI REST    | --> | CrewAI Agents    |
+------------------+     |   API (/v1/chat)  |     |                  |
                         +-------------------+     | 1. Retriever     |
                                                   | 2. Reasoner      |
                                                   | 3. Validator     |
                                                   +------------------+
                                                           |
                              +----------------------------+
                              |                            |
                              v                            v
                    +------------------+        +------------------+
                    | LlamaIndex RAG   |        | Arize Phoenix    |
                    | + PGVector       |        | (Tracing &       |
                    | Retrieval        |        |  Observability)  |
                    +------------------+        +------------------+
                              |
                              v
                    +------------------+
                    | Ollama LLM       |
                    | (llama3.1)       |
                    +------------------+
```

## Component Details

### Ingestion Pipeline
- **Docling**: PDF preprocessing with OCR and table extraction
- **LlamaIndex SentenceSplitter**: Chunking (512 tokens, 50 overlap)
- **Ollama nomic-embed-text**: Embedding generation (768 dimensions)
- **PGVector**: HNSW indexed vector storage

### Agentic RAG Pipeline
- **Agent 1 - Retriever**: Semantic search via PGVector
- **Agent 2 - Reasoner**: Grounded answer generation via Ollama LLM
- **Agent 3 - Validator**: Hallucination detection and quality assurance
- **Process**: Sequential CrewAI workflow

### Observability
- **Arize Phoenix**: Full tracing for LlamaIndex and CrewAI
- **OpenTelemetry**: OTLP span export to Phoenix collector
- **RAGAs**: Automated evaluation (faithfulness, relevancy, precision, recall)

### API Layer
- **FastAPI**: REST API with OpenAPI/Swagger documentation
- **OpenWebUI Integration**: Compatible `/v1/chat/completions` endpoint
- **Background Tasks**: Async ingestion pipeline
