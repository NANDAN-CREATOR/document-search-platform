import logging
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Optional
from pydantic import Field
from ingestion.pgvector_indexer import PGVectorIndexer
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentRetrievalTool(BaseTool):
    name: str = "Document Retrieval Tool"
    description: str = "Retrieves relevant document chunks from PostgreSQL PGVector knowledge base."
    indexer: Optional[PGVectorIndexer] = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indexer = PGVectorIndexer()

    def _run(self, query: str) -> str:
        try:
            retriever = self.indexer.get_retriever(similarity_top_k=settings.similarity_top_k)
            nodes = retriever.retrieve(query)
            if not nodes:
                return "No relevant documents found."
            results = []
            for i, node in enumerate(nodes, 1):
                source = node.metadata.get("filename", "Unknown")
                score = getattr(node, "score", 0.0)
                results.append(f"[Chunk {i}] Source: {source} (Score: {score:.3f})\n{node.get_content()[:500]}...")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return f"Retrieval error: {str(e)}"

def build_crew() -> Crew:
    retrieval_tool = DocumentRetrievalTool()
    retriever_agent = Agent(
        role="Document Retriever",
        goal="Retrieve the most relevant document chunks from PostgreSQL PGVector for the given query.",
        backstory="You are an expert at semantic search using PostgreSQL PGVector.",
        tools=[retrieval_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )
    reasoner_agent = Agent(
        role="Answer Reasoner",
        goal="Generate accurate, grounded answers based on retrieved document context.",
        backstory="You are an expert at reading comprehension and synthesizing information.",
        tools=[],
        verbose=True,
        allow_delegation=False,
    )
    validator_agent = Agent(
        role="Answer Validator",
        goal="Validate the generated answer for accuracy and groundedness.",
        backstory="You are a quality assurance expert who reviews answers for factual grounding.",
        tools=[],
        verbose=True,
        allow_delegation=False,
    )
    return Crew(
        agents=[retriever_agent, reasoner_agent, validator_agent],
        tasks=[],
        process=Process.sequential,
        verbose=True,
    )
