"""Tests for the ValidatorAgent."""
import pytest
from src.agents.validator_agent import ValidatorAgent
from src.agents.state import AgentState

class TestValidatorAgent:
    @pytest.fixture
    def mock_llm(self, mocker):
        mock = mocker.Mock()
        mock.generate.return_value = '{"valid": true, "score": 9, "feedback": "none"}'
        return mock

    @pytest.fixture
    def agent(self, mock_llm):
        return ValidatorAgent(mock_llm, max_attempts=3)

    def test_validator_passes_good_response(self, agent, mock_llm):
        state = AgentState(
            query="What is the policy?",
            session_id="test",
            agent_type="contextual",
            final_answer="The policy is X [1].",
            retrieved_chunks=[{"chunk_text": "The policy is X", "document_name": "doc.pdf", "page_number": 1}]
        )
        result = agent.validate(state)
        
        assert result["valid"] is True
        assert result["score"] == 9
        assert mock_llm.generate.called

    def test_validator_rejects_and_provides_feedback(self, agent, mock_llm):
        mock_llm.generate.return_value = '{"valid": false, "score": 4, "feedback": "Missing citation"}'
        state = AgentState(
            query="What is the policy?",
            session_id="test",
            agent_type="contextual",
            final_answer="The policy is X.", # No citation
            retrieved_chunks=[{"chunk_text": "The policy is X", "document_name": "doc.pdf", "page_number": 1}]
        )
        result = agent.validate(state)
        
        assert result["valid"] is False
        assert result["score"] == 4
        assert "Missing citation" in result["feedback"]

    def test_validator_skips_conversational(self, agent, mock_llm):
        state = AgentState(query="hello", session_id="test", agent_type="conversational")
        result = agent.validate(state)
        
        assert result["valid"] is True
        assert result["score"] == 10
        assert not mock_llm.generate.called # Should skip LLM call entirely