# Presentation Deck — Document Search Platform

> **File:** `document-search-platform.pptx`  
> Agentic RAG with Multi-Agent Pipeline

---

## Slide 1 — Title

**Document Search Platform: Agentic RAG with Multi-Agent Pipeline**

Stack: LlamaIndex · PostgreSQL + PGVector · Ollama · Arize Phoenix · OpenWebUI · FastAPI

*Aman Nandan | Senior Data & AI Engineer | [github.com/NANDAN-CREATOR](https://github.com/NANDAN-CREATOR)*

---

## Slide 2 — The Problem

| Problem | Detail |
|---------|--------|
| 📄 Documents Are Siloed | Critical knowledge locked in PDFs, impossible to query naturally |
| 🤖 LLMs Hallucinate | Without grounding, LLMs generate confident but wrong answers |
| 🔍 Search Is Keyword-Based | Traditional search misses semantic meaning and context |
| 🔗 No Source Attribution | Hard to trace which document an answer came from |

---

## Slide 3 — The Solution

An end-to-end Agentic RAG platform that ingests documents, stores them in a vector database, and uses a 3-agent pipeline to retrieve, reason and validate answers grounded in source documents.

**Pipeline:** PDF Upload → Docling Process → Ollama Embeddings → PGVector Store → RAG Retrieval → LLM Answer

**Key outcomes:**
- ✅ Grounded answers with source citations
- ✅ 3-agent validation pipeline
- ✅ Semantic search via vector embeddings
- ✅ Full observability with Arize Phoenix

---

## Slide 4 — System Architecture

**Ingestion Pipeline**

```
PDF Documents → Docling / PyPDF → LlamaIndex Chunker → Ollama Embeddings → PostgreSQL PGVector
```

**Query Pipeline**

```
OpenWebUI → FastAPI → Agent 1 (Retriever) → Agent 2 (Reasoner) → Agent 3 (Validator)
```

**Cross-cutting:**
- 🔍 Arize Phoenix — Tracing & Observability
- 📊 RAGAs — Faithfulness · Relevancy · Precision · Recall

*Models: Ollama LLM (llama3.2:3b) · Embedding (nomic-embed-text) · Fully local, no cloud dependency*

---

## Slide 5 — Tech Stack

| Tool | Role |
|------|------|
| Docling | PDF preprocessing, OCR, table extraction |
| PostgreSQL + PGVector | Vector database with HNSW index |
| LlamaIndex | RAG framework — chunking, embedding, retrieval |
| Custom 3-Agent System | Retriever → Reasoner → Validator pipeline |
| Ollama | Local LLM provider — llama3.2:3b + nomic-embed-text |
| Arize Phoenix | Inference tracing, PromptOps, observability |
| RAGAs | Faithfulness, relevancy, precision, recall metrics |
| OpenWebUI | Chat frontend — OpenAI-compatible API |
| FastAPI | REST API with Swagger/OpenAPI documentation |

---

## Slide 6 — Multi-Agent RAG Pipeline

| Agent | Responsibilities |
|-------|-----------------|
| **01 · Retriever Agent** | Queries PGVector with semantic search; returns top-k chunks by cosine similarity; formats context for reasoning agent; Tool: `DocumentRetrievalTool` |
| **02 · Reasoner Agent** | Receives query + retrieved context; generates grounded answer via Ollama LLM; uses externalized prompt templates; cites source documents in response |
| **03 · Validator Agent** | Checks answer for hallucinations; validates groundedness in context; ensures completeness and accuracy; returns final validated response |

---

## Slide 7 — Key Features

| Feature | Detail |
|---------|--------|
| 🔍 Semantic Search | HNSW-indexed PGVector enables sub-millisecond similarity search across thousands of document chunks |
| 🤖 Agentic Pipeline | 3-agent sequential workflow with Retriever, Reasoner and Validator for grounded, validated answers |
| 📊 RAG Evaluation | RAGAs metrics: Faithfulness, Answer Relevancy, Context Precision and Recall for continuous quality monitoring |
| 🔭 Full Observability | Arize Phoenix traces every LLM call with latency, token usage and span-level debugging via OpenTelemetry |
| 💬 Chat Interface | OpenWebUI provides a ChatGPT-like interface connected to the document knowledge base |
| ⚡ Local & Private | Fully local deployment — Ollama runs LLMs on your machine, no data sent to external APIs |

---

## Slide 8 — Live Demo Results

**Query:** *"What is RAG architecture?"*

**Answer:**
> The RAG architecture consists of 4 components:
> 1. **Document Ingestion Pipeline:** PDFs preprocessed, embedded using nomic-embed-text, stored in PGVector
> 2. **Retrieval Layer:** Query embedded, semantic similarity search returns top-k chunks
> 3. **Generation Layer:** Retrieved context passed to Ollama LLM for grounded answer generation
> 4. **Evaluation Layer:** RAGAs metrics measure faithfulness, relevancy, precision and recall

**Retrieved sources:** `sample_document.pdf` — scores: 0.619 · 0.455 · 0.419 · 0.413

| Metric | Value |
|--------|-------|
| PDFs Ingested | 1 |
| Chunks Indexed | 4 |
| Top Similarity Score | 0.619 |
| Response Time | < 5s |

---

## Slide 9 — Platform Adaptations (Windows ARM64)

| Tool | Issue | Solution |
|------|-------|----------|
| **Docling** | PyTorch DLL init failure on Windows ARM64 (WinError 1114) | PyPDF used at runtime for text PDFs. Full Docling code in codebase — works on Linux/Mac/x64 Windows |
| **CrewAI** | Dependency chain pulls 500MB+ Google Cloud SDKs + Rust compilation (litellm) | Custom 3-agent system (Retriever → Reasoner → Validator) implements identical architectural pattern using LlamaIndex + Ollama |
| **Arize Phoenix** | sqlean-py requires Microsoft Visual C++ 14.0 build tools | arize-phoenix-otel lightweight client used. Full Phoenix instrumentation code in codebase — runs on Linux/Mac with complete UI |

---

## Slide 10 — Thank You

**Document Search Platform — Fully Working End-to-End**

| Resource | Link |
|----------|------|
| GitHub | [github.com/NANDAN-CREATOR/document-search-platform](https://github.com/NANDAN-CREATOR/document-search-platform) |
| Portfolio | [nandan-creator.github.io/Freelance_Portfolio](https://nandan-creator.github.io/Freelance_Portfolio) |
| API Docs | http://localhost:8000/docs |
| Chat UI | http://localhost:8080 |
