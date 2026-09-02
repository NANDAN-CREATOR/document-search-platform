"""
RAGAs evaluation for the Document Search Platform.

Run:
    python -m evaluation.ragas_eval

Or with custom questions:
    python -m evaluation.ragas_eval --questions-file eval_questions.json
"""
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from config.settings import settings
from agents.retrieval_agent import RetrievalAgent
from agents.rag_pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)

# Default evaluation questions
DEFAULT_QUESTIONS: List[str] = [
    "What is the main topic of the documents?",
    "Summarise the key findings in the knowledge base.",
    "What methodology is described in the documents?",
    "List the important entities mentioned across the documents.",
    "What conclusions or recommendations are made?",
]

DEFAULT_GROUND_TRUTHS: Optional[List[List[str]]] = None


class RAGEvaluator:
    """Evaluate the RAG pipeline using RAGAs metrics."""

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.pipeline = AgenticRAGPipeline()

    def prepare_eval_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[List[str]]] = None,
    ) -> Dataset:
        """Run the pipeline for every question and collect answers + contexts."""
        data: Dict[str, list] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truths": ground_truths if ground_truths else [["N/A"]] * len(questions),
        }

        for idx, question in enumerate(questions, 1):
            logger.info(f"[{idx}/{len(questions)}] Running: {question}")
            try:
                result = self.pipeline.run(question)
                chunks = self.retrieval_agent.retrieve(question)
                contexts = [c["text"] for c in chunks if c.get("text")]
                data["question"].append(question)
                data["answer"].append(result.get("answer", ""))
                data["contexts"].append(contexts if contexts else [""])
            except Exception as exc:
                logger.error(f"Failed for question {idx}: {exc}")
                data["question"].append(question)
                data["answer"].append("")
                data["contexts"].append([""])

        return Dataset.from_dict(data)

    def evaluate_pipeline(
        self,
        questions: List[str],
        ground_truths: Optional[List[List[str]]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate and return a metrics dict; optionally save JSON report."""
        logger.info(f"Starting RAGAs evaluation on {len(questions)} question(s)...")

        dataset = self.prepare_eval_dataset(questions, ground_truths)

        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)
            logger.info("Ground truths provided - including context_recall.")
        else:
            logger.info("No ground truths - skipping context_recall.")

        results = evaluate(dataset=dataset, metrics=metrics)

        output: Dict[str, Any] = {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
        }
        if ground_truths:
            output["context_recall"] = float(results["context_recall"])

        output["num_questions"] = len(questions)
        output["settings"] = {
            "model": settings.ollama_model,
            "embedding_model": settings.ollama_embedding_model,
            "similarity_top_k": settings.similarity_top_k,
            "data_dir": settings.data_dir,
        }

        logger.info(f"RAGAs results: {output}")
        _print_report(output)

        if output_path:
            Path(output_path).write_text(json.dumps(output, indent=2))
            logger.info(f"Report saved to: {output_path}")

        return output


def _print_report(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("  RAGAs Evaluation Report")
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
    parser = argparse.ArgumentParser(description="Run RAGAs evaluation on the Document Search Platform.")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help='Format: {"questions": [...], "ground_truths": [[...], ...]}',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/ragas_report.json",
        help="Path to write the JSON report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
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
