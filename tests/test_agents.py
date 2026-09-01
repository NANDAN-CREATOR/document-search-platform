import pytest
from unittest.mock import patch, MagicMock

def test_retrieval_agent_format_context_empty():
    from agents.retrieval_agent import RetrievalAgent
    with patch("agents.retrieval_agent.PGVectorIndexer"):
        agent = RetrievalAgent()
        result = agent.format_context([])
        assert result == "No relevant context found."

def test_retrieval_agent_format_context():
    from agents.retrieval_agent import RetrievalAgent
    with patch("agents.retrieval_agent.PGVectorIndexer"):
        agent = RetrievalAgent()
        chunks = [{"text": "Test content", "source": "test.pdf", "score": 0.9, "metadata": {}}]
        result = agent.format_context(chunks)
        assert "Source 1" in result
        assert "test.pdf" in result
        assert "Test content" in result
