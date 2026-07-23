"""Tests for the ResearchAgent."""
import pytest
from src.agents.research_agent import ResearchAgent
from src.agents.state import AgentState

class TestResearchAgent:
    @pytest.fixture
    def mock_llm(self, mocker):
        mock = mocker.Mock()
        mock.generate.return_value = "The policy changed in 2020 [1][2]."
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        mock = mocker.Mock()
        mock.retrieve.return_value = [{
            "chunk_text": "Policy changed in 2020", 
            "document_name": "pol.pdf", 
            "page_number": 1
        }]
        return mock

    @pytest.fixture
    def agent(self, mock_retriever, mock_llm, mocker):
        # Mock the WebSearchTool instantiation inside ResearchAgent
        mock_web_search = mocker.Mock()
        mock_web_search.search.return_value = [{
            "title": "News Article", 
            "url": "http://test.com", 
            "content": "Policy changed"
        }]
        mocker.patch("src.agents.research_agent.WebSearchTool", return_value=mock_web_search)
        return ResearchAgent(mock_retriever, mock_llm, web_search_enabled=True)

    def test_research_agent_combines_sources(self, agent, mock_retriever, mock_llm):
        state = AgentState(query="test query", session_id="test", web_search_enabled=True)
        result = agent.run(state)
        
        assert mock_retriever.retrieve.called
        assert mock_llm.generate.called
        assert len(result.citations) >= 1
        assert "research_agent" in result.agent_path

    def test_research_agent_handles_no_results(self, mock_llm, mocker):
        mock_retriever = mocker.Mock()
        mock_retriever.retrieve.return_value = []
        
        mock_web_search = mocker.Mock()
        mock_web_search.search.return_value = []
        mocker.patch("src.agents.research_agent.WebSearchTool", return_value=mock_web_search)
        
        agent = ResearchAgent(mock_retriever, mock_llm, web_search_enabled=True)
        state = AgentState(query="test", session_id="test", web_search_enabled=True)
        result = agent.run(state)
        
        assert "not found" in result.final_answer.lower()