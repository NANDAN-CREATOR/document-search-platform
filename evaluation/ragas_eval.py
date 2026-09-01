import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from config.settings import settings
from agents.retrieval_agent import RetrievalAgent
from agents.rag_pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.pipeline = AgenticRAGPipeline()

    def prepare_eval_dataset(self, questions: List[str], ground_truths: Optional[List[List[str]]] = None) -> Dataset:
        data = {"question": [], "answer": [], "contexts": [], "ground_truths": ground_truths or [["N/A"]] * len(questions)}
        for question in questions:
            result = self.pipeline.run(question)
            chunks = self.retrieval_agent.retrieve(question)
            contexts = [c["text"] for c in chunks]
            data["question"].append(question)
            data["answer"].append(result["answer"])
            data["contexts"].append(contexts)
        return Dataset.from_dict(data)

    def evaluate_pipeline(self, questions: List[str], ground_truths: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        logger.info(f"Evaluating RAG pipeline on {len(questions)} questions...")
        dataset = self.prepare_eval_dataset(questions, ground_truths)
        results = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        metrics = {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
            "context_recall": float(results["context_recall"]),
        }
        logger.info(f"Evaluation results: {metrics}")
        return metrics
