from src.agents.state import AgentState
from src.utils.logger import logger

class ConversationalAgent:
    """Handles greetings, small talk, and general knowledge"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, state: AgentState) -> AgentState:
        """Generate conversational response"""
        logger.info(f"ConversationalAgent processing: {state.query}")
        
        prompt = f"""You are a friendly AI assistant. Respond naturally to the user's message.

User: {state.query}

Assistant:"""
        
        answer = self.llm.generate(prompt)
        
        state.final_answer = answer
        state.agent_type = "conversational"
        state.agent_path.append("conversational_agent")
        
        return state