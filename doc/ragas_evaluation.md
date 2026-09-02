# RAGAs Evaluation

This document covers the RAG evaluation methodology, metrics used, and how to run the evaluation pipeline against the Document Search Platform.

---

## Overview

[RAGAs](https://ragas.io) (Retrieval Augmented Generation Assessment) is used to measure the quality of the RAG pipeline across four key dimensions. Evaluation is implemented in [`evaluation/ragas_eval.py`](../evaluation/ragas_eval.py).

---

## Metrics

| Metric | Requires Ground Truth | Description |
|--------|----------------------|-------------|
| **Faithfulness** | No | Measures whether the generated answer is factually grounded in the retrieved context. Score: 0–1 (higher = more faithful) |
| **Answer Relevancy** | No | Measures how relevant the answer is to the question asked. Penalises incomplete or redundant answers. Score: 0–1 |
| **Context Precision** | No | Measures whether the retrieved chunks are relevant to the question. High precision = no noise in context. Score: 0–1 |
| **Context Recall** | **Yes** | Measures whether all information needed to answer is present in the retrieved context. Requires reference answers. Score: 0–1 |

> **Note:** Faithfulness, Answer Relevancy and Context Precision run without ground truths (reference-free). Context Recall requires ground truth answers per question.

---

## How to Run

### 1. Default evaluation (no ground truths needed)

```bash
python -m evaluation.ragas_eval
```

Runs 5 built-in domain-agnostic questions. Produces a terminal report and saves `evaluation/ragas_report.json`.

### 2. Custom questions file

Create `eval_questions.json`:

```json
{
  "questions": [
    "What is the main topic of the documents?",
    "What methodology is described?",
    "List the key entities mentioned."
  ],
  "ground_truths": [
    ["The documents cover ..."],
    ["The methodology involves ..."],
    ["Key entities include ..."]
  ]
}
```

Then run:

```bash
python -m evaluation.ragas_eval --questions-file eval_questions.json --output evaluation/ragas_report.json
```

### 3. Custom output path

```bash
python -m evaluation.ragas_eval --output results/my_eval.json
```

---

## Output

### Terminal report

```
=======================================================
  RAGAs Evaluation Report
=======================================================
  faithfulness           0.8750  [#################   ]
  answer_relevancy       0.9120  [##################  ]
  context_precision      0.8400  [#################   ]
  context_recall         0.7900  [###############     ]

  Questions evaluated : 5
  Model               : llama3.1
  Embedding           : nomic-embed-text
  Top-K               : 5
  Data dir            : ./data
=======================================================
```

### JSON report (`evaluation/ragas_report.json`)

```json
{
  "faithfulness": 0.875,
  "answer_relevancy": 0.912,
  "context_precision": 0.84,
  "context_recall": 0.79,
  "num_questions": 5,
  "settings": {
    "model": "llama3.1",
    "embedding_model": "nomic-embed-text",
    "similarity_top_k": 5,
    "data_dir": "./data"
  }
}
```

---

## Prerequisites

Ensure the following are running before evaluation:

```bash
# 1. Ollama with required models
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. PostgreSQL + PGVector (via Docker)
docker-compose up -d postgres

# 3. Documents ingested
python -m ingestion.pipeline

# 4. Install eval dependencies
pip install ragas datasets
```

---

## Interpreting Results

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 – 1.0 | Excellent — pipeline is highly accurate and well-grounded |
| 0.7 – 0.9 | Good — minor hallucinations or context gaps; review chunking strategy |
| 0.5 – 0.7 | Fair — consider tuning chunk size, top-k, or prompt templates |
| < 0.5 | Poor — investigate retrieval quality, embedding model, or document quality |

### Improving scores

- **Low faithfulness** → Tighten the system prompt in `prompts/system_prompt.yaml` to enforce "answer only from context"
- **Low answer relevancy** → Refine the reasoner agent's prompt; ensure questions are domain-specific
- **Low context precision** → Reduce `similarity_top_k` or increase chunk overlap
- **Low context recall** → Increase `similarity_top_k`; check that all relevant PDFs are ingested

---

## Integration with Arize Phoenix

All RAG pipeline calls during evaluation are automatically traced by Arize Phoenix (when running). View traces at http://localhost:6006 to inspect per-question retrieval quality, LLM latency, and span-level debugging.
