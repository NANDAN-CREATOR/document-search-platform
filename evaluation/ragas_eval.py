"""
RAGAs evaluation for the Document Search Platform.

Run:
    python -m evaluation.ragas_eval
"""
import sys
import types
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def _stub_module(name: str, **attrs):
    """Inject a dummy module into sys.modules if not already present."""
    if name not in sys.modules:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m


def _patch_ragas_deps():
    """
    Ragas <0.2 relies on langchain_community/langcore internals that
    newer langchain versions removed. We stub them out so ragas can
    import without crashing. No package is installed or changed.
    """
    # Stub parent modules first
    _stub_module("langchain_community")
    _stub_module("langchain_community.chat_models")
    _stub_module("langchain_community.chat_models.vertexai", ChatVertexAI=object)

    # Stub langchain_core.pydantic_v1 (removed in langchain-core 0.3+)
    try:
        from langchain_core import pydantic_v1  # noqa: F401
    except ImportError:
        try:
            # Try to point at pydantic v1 compat layer if available
            import pydantic.v1 as _pydv1
            _stub_module("langchain_core.pydantic_v1",
                         BaseModel=_pydv1.BaseModel,
                         Field=_pydv1.Field,
                         root_validator=getattr(_pydv1, 'model_validator', lambda *a, **k: lambda f: f),
                         validator=getattr(_pydv1, 'field_validator', lambda *a, **k: lambda f: f))
        except ImportError:
            # Fall back to plain pydantic v2
            import pydantic as _pyd
            def _noop_dec(*a, **k):
                return lambda f: f
            _stub_module("langchain_core.pydantic_v1",
                         BaseModel=._pyd.BaseModel,
                         Field=_pyd.Field,
                         root_validator=_noop_dec,
                         validator=_noop_dec)


_patch_ragas_deps()

# Now safe to import ragas
from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config.settings import settings
from agents.retrieval_agent import RetrievalAgent
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
        logger.info(f"Starting RAGAs evaluation on {len(questions)} question(s)...")
        dataset = self.prepare_eval_dataset(questions, ground_truths)
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)
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
    parser = argparse.ArgumentParser(description="Run RAGAs evaluation.")
    parser.add_argument("--questions-file", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation/ragas_report.json")
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
