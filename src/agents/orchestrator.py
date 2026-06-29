from src.agents.state import AgentState
from src.agents.router_agent import RouterAgent
from src.agents.qa_agent import QAAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.critic_agent import CriticAgent
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.ollama_client import OllamaClient
from src.agents.researcher_agent import ResearcherAgent
from src.utils.logger import logger

class AgentOrchestrator:
    """Coordinates multi-agent workflow with self-critique."""
    
    def __init__(self, retriever: RetrievalPipeline, llm: OllamaClient):
        self.router = RouterAgent(llm)
        self.qa_agent = QAAgent(retriever, llm)
        self.analyst_agent = AnalystAgent(retriever, llm)
        self.critic = CriticAgent(llm, max_retries=1)
        self.llm = llm
        self.researcher_agent = ResearcherAgent(retriever, llm)

    def run(self, query: str, session_id: str = "default") -> AgentState:
        """Execute the multi-agent pipeline with critique loop."""
        
        state = AgentState(query=query, session_id=session_id)
        
        # Step 1: Route query
        state = self.router.run(state)
        logger.info(f"Query classified as: {state.query_type}")
        
        # Step 2: Execute specialist agent
        state = self._run_specialist(state)
        
        # Save the first answer in case retry makes it worse
        first_answer = state.final_answer
        first_citations = state.citations.copy()
        
        # Step 3: Self-critique loop
        state = self.critic.run(state)
        
        # Step 4: Retry if critique failed
        if state.critique_failed and state.retry_count < self.critic.max_retries:
            logger.info(f"Retrying with stricter prompt (attempt {state.retry_count + 1})")
            state.retry_count += 1
            state = self._run_specialist(state, strict=True)
            state = self.critic.run(state)
            
            # ✅ If retry made things worse, revert to first answer
            if state.critique_failed:
                logger.warning("Retry failed, reverting to original answer")
                state.final_answer = first_answer
                state.citations = first_citations
        
        logger.info(f"Final agent path: {' -> '.join(state.agent_path)}")
        return state
    
    def _run_specialist(self, state: AgentState, strict: bool = False) -> AgentState:
        """Dispatch to the appropriate specialist agent."""
    
        if state.query_type == "qa":
            return self.qa_agent.run(state)
        elif state.query_type == "analysis":
            return self.analyst_agent.run(state)
        elif state.query_type == "research":
            return self.researcher_agent.run(state)
        
        return state