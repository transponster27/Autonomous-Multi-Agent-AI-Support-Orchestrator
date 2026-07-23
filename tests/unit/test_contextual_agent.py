"""Tests for the ContextualAgent."""

import pytest
from src.agents.contextual_agent import ContextualAgent
from src.agents.state import AgentState


class TestContextualAgent:
    """Test suite for ContextualAgent"""
    
    @pytest.fixture
    def mock_llm(self, mocker):
        """Mock LLM that returns a grounded answer."""
        mock = mocker.Mock()
        mock.generate.return_value = "Employees get 15 days of leave [1]."
        return mock
    
    @pytest.fixture
    def mock_retriever(self, mocker):
        """Mock retriever that returns fake chunks."""
        mock = mocker.Mock()
        mock.retrieve.return_value = [
            {
                "chunk_id": "chunk_0_policy.pdf",
                "document_name": "policy.pdf",
                "page_number": 3,
                "chunk_text": "Employees are entitled to 15 days of paid annual leave."
            },
            {
                "chunk_id": "chunk_1_handbook.pdf",
                "document_name": "handbook.pdf",
                "page_number": 12,
                "chunk_text": "Leave can be carried forward up to 5 days."
            }
        ]
        return mock
    
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        """Create ContextualAgent with mocks."""
        return ContextualAgent(mock_retriever, mock_llm)
    
    def test_agent_calls_retriever(self, agent, mock_retriever):
        """Agent should call retriever.retrieve()."""
        state = AgentState(query="What is the leave policy?", session_id="test")
        agent.run(state)
        
        assert mock_retriever.retrieve.called
        # Check it was called with the query
        call_args = mock_retriever.retrieve.call_args
        assert call_args[0][0] == "What is the leave policy?"
    
    def test_agent_calls_llm_with_context(self, agent, mock_llm, mock_retriever):
        """Agent should call LLM with retrieved context."""
        state = AgentState(query="What is the leave policy?", session_id="test")
        agent.run(state)
        
        assert mock_llm.generate.called
        prompt = mock_llm.generate.call_args[0][0]
        
        # Prompt should contain the query
        assert "What is the leave policy?" in prompt
        # Prompt should contain retrieved text
        assert "15 days of paid annual leave" in prompt
    
    def test_agent_returns_citations(self, agent):
        """Agent should build numbered citations."""
        state = AgentState(query="What is the leave policy?", session_id="test")
        result = agent.run(state)
        
        assert len(result.citations) > 0
        # Check citation structure
        citation = result.citations[0]
        assert "number" in citation
        assert "document" in citation
        assert "page" in citation
    
    def test_agent_handles_no_results(self, mock_llm, mocker):
        """Agent should handle case where retriever returns nothing."""
        mock_retriever = mocker.Mock()
        mock_retriever.retrieve.return_value = []
        
        agent = ContextualAgent(mock_retriever, mock_llm)
        state = AgentState(query="What is X?", session_id="test")
        result = agent.run(state)
        
        assert "not found" in result.final_answer.lower()
    
    def test_agent_sets_documents_searched(self, agent):
        """Agent should track which documents were searched."""
        state = AgentState(query="What is the leave policy?", session_id="test")
        result = agent.run(state)
        
        assert len(result.documents_searched) > 0
        assert "policy.pdf" in result.documents_searched