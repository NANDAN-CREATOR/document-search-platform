import logging
from crewai import Crew, Process, Task
from agents.crew_config import build_crew
from agents.retrieval_agent import RetrievalAgent
from tracing.phoenix_setup import instrument_crewai
from config.settings import settings

logger = logging.getLogger(__name__)

class AgenticRAGPipeline:
    def __init__(self):
        instrument_crewai()
        self.retrieval_helper = RetrievalAgent()

    def run(self, query: str) -> dict:
        logger.info(f"Running Agentic RAG for query: '{query[:80]}'")
        retrieved_chunks = self.retrieval_helper.retrieve(query)
        context = self.retrieval_helper.format_context(retrieved_chunks)
        crew = build_crew()
        retriever_agent, reasoner_agent, validator_agent = crew.agents

        retrieval_task = Task(
            description=f"Find relevant information for: {query}",
            expected_output="Relevant document chunks with sources.",
            agent=retriever_agent,
        )
        reasoning_task = Task(
            description=f"Using this context, answer: {query}\n\nContext:\n{context}",
            expected_output="A grounded, cited answer.",
            agent=reasoner_agent,
            context=[retrieval_task],
        )
        validation_task = Task(
            description=f"Validate the answer to: {query}\nContext: {context[:500]}",
            expected_output="Validation report confirming accuracy.",
            agent=validator_agent,
            context=[reasoning_task],
        )
        dynamic_crew = Crew(
            agents=[retriever_agent, reasoner_agent, validator_agent],
            tasks=[retrieval_task, reasoning_task, validation_task],
            process=Process.sequential,
            verbose=True,
        )
        result = dynamic_crew.kickoff()
        return {
            "query": query,
            "answer": str(result),
            "sources": [{"filename": c["source"], "score": c["score"]} for c in retrieved_chunks],
            "chunks_retrieved": len(retrieved_chunks),
        }
