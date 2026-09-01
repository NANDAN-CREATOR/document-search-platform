import logging
from typing import List
from crewai import Task
from ingestion.pgvector_indexer import ChromaIndexer
from prompts.prompt_manager import get_prompt
from config.settings import settings

logger = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self):
        self.indexer = ChromaIndexer()

    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        top_k = top_k or settings.similarity_top_k
        retriever = self.indexer.get_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        results = []
        for node in nodes:
            results.append({
                "text": node.get_content(),
                "score": getattr(node, "score", 0.0),
                "source": node.metadata.get("filename", "Unknown"),
                "metadata": node.metadata,
            })
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}'")
        return results

    def format_context(self, retrieved_chunks: List[dict]) -> str:
        if not retrieved_chunks:
            return "No relevant context found."
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(context_parts)

    def build_retrieval_task(self, agent, query: str) -> Task:
        return Task(
            description=get_prompt("retrieval_prompt", query=query),
            expected_output="A list of the most relevant document chunks with source filenames.",
            agent=agent,
        )
