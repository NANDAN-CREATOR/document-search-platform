import logging
from crewai import Task
from prompts.prompt_manager import get_prompt

logger = logging.getLogger(__name__)

class ReasoningAgent:
    def build_reasoning_task(self, agent, context: str, question: str) -> Task:
        return Task(
            description=get_prompt("reasoning_prompt", context=context, question=question),
            expected_output="A clear, concise, factually grounded answer with source citations.",
            agent=agent,
        )

    def build_validation_task(self, agent, context: str, question: str, answer: str) -> Task:
        return Task(
            description=get_prompt("validation_prompt", context=context, question=question, answer=answer),
            expected_output="A validation report confirming accuracy and groundedness.",
            agent=agent,
        )
