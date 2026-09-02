"""
Standalone RAG Evaluation -- no ragas package required.
Uses Ollama LLM directly to score the same 4 metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall (when ground truths provided)

Run:
    python -m evaluation.ragas_eval
"""
import json
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

from config.settings import settings
from agents.rag_pipeline import RetrieverAgent as RetrievalAgent
from agents.rag_pipeline import AgenticRAGPipeline
logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS: List[str] = [
    "What is the main topic of the documents?",
    "Summarise the key findings in the knowledge base.",
    "What methodology is described in the documents?",
    "List the important entities mentioned across the documents.",
    "What conclusions or recommendations are made?",
]

DEFAULT_GROUND_TRUTHS: Optional[List[List[str]]] = None


# -----------------------------------------------------------------------
# Ollama LLM scorer
# -----------------------------------------------------------------------
def _llm_score(prompt: str) -> float:
    """Send a scoring prompt to Ollama and extract a 0.0-1.0 score."""
    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        text = resp.json().get("response", "0").strip()
        match = re.search(r'\d+(?:\.\d+)?', text)
        if match:
            score = float(match.group())
            return min(max(score, 0.0), 1.0)
        return 0.0
    except Exception as e:
        logger.warning(f"LLM scoring failed: {e}")
        return 0.0


# -----------------------------------------------------------------------
# Metric scorers
# -----------------------------------------------------------------------
def score_faithfulness(answer: str, contexts: List[str]) -> float:
    context_block = "\n\n".join(contexts)[:2000]
    prompt = f"""You are a RAG evaluator. Score whether the ANSWER is faithful to the CONTEXT (no hallucinations).
Respond with ONLY a single number between 0 and 1.
1.0 = fully grounded, 0.0 = completely ungrounded.

CONTEXT:
{context_block}

ANSWER:
{answer[:1000]}

Score:"""
    return _llm_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""You are a RAG evaluator. Score how relevant the ANSWER is to the QUESTION.
Respond with ONLY a single number between 0 and 1.
1.0 = perfectly relevant and complete, 0.0 = completely irrelevant.

QUESTION: {question}
ANSWER: {answer[:1000]}

Score:"""
    return _llm_score(prompt)


def score_context_precision(question: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
    scores = []
    for chunk in contexts[:5]:
        prompt = f"""You are a RAG evaluator. Score how relevant this CONTEXT CHUNK is to answering the QUESTION.
Respond with ONLY a single number between 0 and 1.

QUESTION: {question}
CHUNK: {chunk[:500]}

Score:"""
        scores.append(_llm_score(prompt))
    return sum(scores) / len(scores)


def score_context_recall(contexts: List[str], ground_truth: str) -> float:
    context_block = "\n\n".join(contexts)[:2000]
    prompt = f"""You are a RAG evaluator. Score whether the CONTEXT contains enough information to support the GROUND TRUTH.
Respond with ONLY a single number between 0 and 1.

CONTEXT:
{context_block}

GROUND TRUTH: {ground_truth}

Score:"""
    return _llm_score(prompt)


# -----------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------
class RAGEvaluator:
    """Evaluate the RAG pipeline using Ollama LLM as scorer."""

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.pipeline = AgenticRAGPipeline()

    def evaluate_pipeline(
        self,
        questions: List[str],
        ground_truths: Optional[List[List[str]]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Starting RAG evaluation on {len(questions)} question(s)...")

        all_faithfulness = []
        all_relevancy = []
        all_precision = []
        all_recall = []
        per_question = []

        for idx, question in enumerate(questions, 1):
            logger.info(f"[{idx}/{len(questions)}] {question}")
            try:
                result = self.pipeline.run(question)
                answer = result.get("answer", "")
                chunks = self.retrieval_agent.run(question)
                contexts = [c["text"] for c in chunks if c.get("text")]

                logger.info("  Scoring faithfulness...")
                f = score_faithfulness(answer, contexts)
                logger.info("  Scoring answer relevancy...")
                ar = score_answer_relevancy(question, answer)
                logger.info("  Scoring context precision...")
                cp = score_context_precision(question, contexts)

                row = {
                    "question": question,
                    "answer": answer[:300],
                    "chunks_retrieved": len(chunks),
                    "faithfulness": round(f, 4),
                    "answer_relevancy": round(ar, 4),
                    "context_precision": round(cp, 4),
                }

                if ground_truths:
                    gt = " ".join(ground_truths[idx - 1])
                    logger.info("  Scoring context recall...")
                    cr = score_context_recall(contexts, gt)
                    row["context_recall"] = round(cr, 4)
                    all_recall.append(cr)

                all_faithfulness.append(f)
                all_relevancy.append(ar)
                all_precision.append(cp)
                per_question.append(row)
                logger.info(f"  F={f:.3f} AR={ar:.3f} CP={cp:.3f}")

            except Exception as exc:
                logger.error(f"Failed for question {idx}: {exc}")

        def _avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        output: Dict[str, Any] = {
            "faithfulness": _avg(all_faithfulness),
            "answer_relevancy": _avg(all_relevancy),
            "context_precision": _avg(all_precision),
            "num_questions": len(questions),
            "per_question": per_question,
            "settings": {
                "model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "similarity_top_k": settings.similarity_top_k,
                "data_dir": settings.data_dir,
            },
        }
        if all_recall:
            output["context_recall"] = _avg(all_recall)

        _print_report(output)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(output, indent=2))
            logger.info(f"Report saved to: {output_path}")

        return output


def _print_report(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("  RAG Evaluation Report (Ollama-scored)")
    print("=" * 55)
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if key in metrics:
            bar = "#" * int(metrics[key] * 20)
            print(f"  {key:<22} {metrics[key]:.4f}  [{bar:<20}]")
    print(f"\n  Questions evaluated : {metrics.get('num_questions', '?')}")
    cfg = metrics.get("settings", {})
    print(f"  Model               : {cfg.get('model', '?')}")
    print(f"  Embedding           : {cfg.get('embedding_model', '?')}")
    print(f"  Top-K               : {cfg.get('similarity_top_k', '?')}")
    print(f"  Data dir            : {cfg.get('data_dir', '?')}")
    print("=" * 55 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation using Ollama scorer.")
    parser.add_argument("--questions-file", type=str, default=None,
                        help='JSON file: {"questions": [...], "ground_truths": [[...]]}')
    parser.add_argument("--output", type=str, default="evaluation/ragas_report.json",
                        help="Path to write JSON report")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    args = _parse_args()
    questions = DEFAULT_QUESTIONS
    ground_truths = DEFAULT_GROUND_TRUTHS
    if args.questions_file:
        data = json.loads(Path(args.questions_file).read_text())
        questions = data["questions"]
        ground_truths = data.get("ground_truths")
    RAGEvaluator().evaluate_pipeline(
        questions=questions,
        ground_truths=ground_truths,
        output_path=args.output,
    )
