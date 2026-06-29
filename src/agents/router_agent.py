from src.agents.state import AgentState
from src.llm.ollama_client import OllamaClient

class RouterAgent:
    """Classifies queries and routes to specialist agents"""
    
    def __init__(self, llm: OllamaClient):
        self.llm = llm
    
    def run(self, state: AgentState) -> AgentState:
        """Classify query and update state"""
    
        prompt = f"""Classify this query into EXACTLY ONE of these three categories:

    1. "qa" - Simple factual question asking for a definition, fact, or specific detail
    Examples:
    - "What is business policy?"
    - "Who is responsible for policy formulation?"
    - "What are the features of a good policy?"
    - "When was the company founded?"

    2. "analysis" - Requires comparison, summary, or synthesis of multiple points
    Examples:
    - "Compare policy and strategy"
    - "Summarize the main features"
    - "What are the differences between X and Y?"
    - "List all the components mentioned"

    3. "research" - Complex question requiring multi-step reasoning or deep investigation
    Examples:
    - "How do the components of strategic management relate to business policy?"
    - "Analyze the relationship between X and Y across all documents"
    - "What are the implications of Z for the entire organization?"

    Query: {state.query}

    Respond with ONLY the category name (qa, analysis, or research):"""

        classification = self.llm.generate(prompt).strip().lower()
        
        if classification not in ["qa", "analysis", "research"]:
            classification = "qa"
        
        state.query_type = classification
        state.agent_path.append("router")
        
        return state