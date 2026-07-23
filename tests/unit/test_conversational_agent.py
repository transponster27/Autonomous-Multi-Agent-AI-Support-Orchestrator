"""Tests for the ConversationalAgent."""

import pytest
from src.agents.conversational_agent import ConversationalAgent
from src.agents.state import AgentState


class TestConversationalAgent:
    """Test suite for ConversationalAgent"""
    
    @pytest.fixture
    def mock_llm(self, mocker):
        """Create a mock LLM that returns predictable responses."""
        mock = mocker.Mock()
        mock.generate.return_value = "Hello! How can I help you today?"
        return mock
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create a ConversationalAgent with mock LLM."""
        return ConversationalAgent(mock_llm)
    
    def test_agent_sets_agent_type(self, agent):
        """Agent should set agent_type to 'conversational'."""
        state = AgentState(query="hello", session_id="test")
        result = agent.run(state)
        
        assert result.agent_type == "conversational"
    
    def test_agent_calls_llm(self, agent, mock_llm):
        """Agent should call LLM.generate() with the query."""
        state = AgentState(query="hi there", session_id="test")
        agent.run(state)
        
        # Verify LLM was called
        assert mock_llm.generate.called
        # Check the prompt contains the query
        call_args = mock_llm.generate.call_args[0][0]
        assert "hi there" in call_args
    
    def test_agent_returns_final_answer(self, agent, mock_llm):
        """Agent should populate final_answer in state."""
        state = AgentState(query="hello", session_id="test")
        result = agent.run(state)
        
        assert result.final_answer is not None
        assert result.final_answer == "Hello! How can I help you today?"
    
    def test_agent_appends_to_path(self, agent):
        """Agent should record itself in agent_path."""
        state = AgentState(query="hello", session_id="test")
        result = agent.run(state)
        
        assert "conversational_agent" in result.agent_path
    
    def test_agent_handles_greetings(self, agent, mock_llm):
        """Agent should respond to various greetings."""
        greetings = ["hello", "hi", "hey", "good morning"]
        
        for greeting in greetings:
            state = AgentState(query=greeting, session_id="test")
            result = agent.run(state)
            
            assert result.final_answer is not None
            assert result.agent_type == "conversational"