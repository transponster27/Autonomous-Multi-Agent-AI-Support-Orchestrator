from src.agents.state import AgentState
from src.agents.conversational_agent import ConversationalAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.research_agent import ResearchAgent
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.ollama_client import OllamaClient
from src.utils.logger import logger
from typing import List, Optional

class AgentOrchestrator:
    def __init__(self, retriever: RetrievalPipeline, llm: OllamaClient):
        self.conversational = ConversationalAgent(llm)
        self.contextual = ContextualAgent(retriever, llm)
        self.retriever = retriever
        self.llm = llm
    
    def run(
        self, 
        query: str, 
        session_id: str = "default", 
        document_filter: Optional[List[str]] = None, 
        domain: Optional[str] = None,
        web_search_enabled: bool = False
    ) -> AgentState:

        print(f"ORCHESTRATOR: web_search_enabled = {web_search_enabled}")
        
        state = AgentState(
            query=query,
            session_id=session_id,
            document_filter=document_filter or [],
            domain=domain,
            web_search_enabled=web_search_enabled
        )
        
        # Log the state for debugging
        logger.info(f"Orchestrator received - web_search_enabled: {state.web_search_enabled}")
        
        # Route query
        state = self._route_query(state)
        
        logger.info(f"Routed to agent type: {state.agent_type}")
        
        # Execute appropriate agent
        if state.agent_type == "conversational":
            state = self.conversational.run(state)
        elif state.agent_type == "research":
            # Initialize ResearchAgent with web search flag
            research_agent = ResearchAgent(self.retriever, self.llm, web_search_enabled)
            state = research_agent.run(state)
        else:
            state = self.contextual.run(state)
        
        logger.info(f"Agent path: {' -> '.join(state.agent_path)}")
        return state
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Route query to appropriate agent based on content and flags"""
        
        # If web search is explicitly enabled, always route to research
        if state.web_search_enabled:
            state.agent_type = "research"
            state.agent_path.append("router")
            logger.info("Web search enabled - routing to research agent")
            return state
        
        # Three-category classification
        prompt = f"""Classify this query into exactly one category.

Query: {state.query}

Category "conversational" - Use ONLY for:
- Greetings: hello, hi, hey, good morning, good evening
- Pleasantries: thanks, thank you, bye, goodbye, see you
- Questions about the assistant: who are you, what can you do, how are you
- Small talk: how's it going, what's up (as greeting)

Category "research" - Use for complex queries requiring deep analysis:
- Requests for comprehensive research or investigation
- Queries asking for "deep dive", "extensive research", "thorough analysis"
- Questions requiring multiple sources or perspectives
- Complex topics needing detailed exploration (e.g., "journey of an AI engineer", "complete guide to...")
- When the query explicitly mentions "research", "investigate", or "analyze deeply"

Category "contextual" - Use for straightforward information requests:
- Simple factual questions
- Requests for definitions, explanations, or summaries
- Questions about specific documents or policies

Respond with ONLY the word "conversational", "research", or "contextual":"""

        decision = self.llm.generate(prompt).strip().lower()
        
        if "conversational" in decision:
            state.agent_type = "conversational"
        elif "research" in decision:
            state.agent_type = "research"
        else:
            state.agent_type = "contextual"
        
        state.agent_path.append("router")
        logger.info(f"LLM classified query as: {state.agent_type}")
        return state