from src.agents.state import AgentState
from src.agents.conversational_agent import ConversationalAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.research_agent import ResearchAgent
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.ollama_client import OllamaClient
from src.agents.validator_agent import ValidatorAgent 
from src.utils.logger import logger
from typing import List, Optional
import re


class AgentOrchestrator:
    def __init__(self, retriever: RetrievalPipeline, llm: OllamaClient):
        self.conversational = ConversationalAgent(llm)
        self.contextual = ContextualAgent(retriever, llm)
        self.retriever = retriever
        self.llm = llm
        self.validator = ValidatorAgent(llm, max_attempts=3)
    
    def run(
        self, 
        query: str, 
        session_id: str = "default", 
        document_filter: Optional[List[str]] = None, 
        domain: Optional[str] = None,
        web_search_enabled: bool = False
    ) -> AgentState:

        logger.debug(f"ORCHESTRATOR: web_search_enabled = {web_search_enabled}")
        
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
        
        # Run validation loop for non-conversational responses
        if state.agent_type != "conversational":
            state = self._validate_and_retry(state)
        
        logger.info(f"Agent path: {' -> '.join(state.agent_path)}")
        return state
    
    def _validate_and_retry(self, state: AgentState) -> AgentState:
        """
        Validation loop: critic reviews the answer, and if invalid,
        regenerates with feedback. Max 3 attempts.
        """
        for attempt in range(1, self.validator.max_attempts + 1):
            state.validation_attempts = attempt
            
            # Critic evaluates the current answer
            validation = self.validator.validate(state)
            state.validation_score = validation["score"]
            state.validation_feedback = validation.get("feedback", "")
            
            logger.info(
                f"Validation attempt {attempt}/{self.validator.max_attempts} - "
                f"Score: {validation['score']}/10 - Valid: {validation['valid']}"
            )
            
            # If valid, we're done
            if validation["valid"]:
                state.agent_path.append("validator_agent(passed)")
                return state
            
            # If invalid and we have attempts left, regenerate
            if attempt < self.validator.max_attempts:
                logger.warning(f"Regenerating with feedback: {validation.get('feedback')}")
                state = self._regenerate_with_feedback(state, validation.get("feedback", ""))
                state.agent_path.append(f"regeneration_{attempt}")
            else:
                # Max attempts reached — use last answer anyway
                logger.warning("Max validation attempts reached. Using final response.")
                state.agent_path.append("validator_agent(max_attempts)")
        
        return state
    
    def _regenerate_with_feedback(self, state: AgentState, feedback: str) -> AgentState:
        """Regenerate the answer incorporating critic feedback"""
        
        # Build numbered context
        context = ""
        for i, chunk in enumerate(state.retrieved_chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            text = chunk.get("chunk_text", "")
            context += f"[{i}] {doc_name} (Page {page}):\n{text}\n\n"
        
        prompt = f"""The previous answer was rejected by a quality validator.
Regenerate it addressing ALL the feedback below.

Question: {state.query}

Sources:
{context}
Validator Feedback:
{feedback}

STRICT RULES:
1. Start DIRECTLY with the answer. NO preamble.
2. Use inline [1], [2] citations after each claim.
3. Do NOT include author full names.
4. Address every issue mentioned in the feedback.
5. If information is not in sources, say "Information not found."

Regenerated Answer (start directly):"""
        
        answer = self.llm.generate(prompt)
        state.final_answer = self._clean_preamble(answer)
        return state
    
    def _clean_preamble(self, text: str) -> str:
        patterns = [
            r"^(Based on the provided (documents|sources|context|information)[,.]?\s*)",
            r"^(According to the (documents|sources)[,.]?\s*)",
            r"^(The (documents|sources) (state|indicate|show)[,.]?\s*)",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text.strip()
     
    def _route_query(self, state: AgentState) -> AgentState:
        """Route query to appropriate agent based on content and flags"""
        
        # If web search is explicitly enabled, always route to research
        if state.web_search_enabled:
            state.agent_type = "research"
            state.agent_path.append("router")
            logger.info("Web search enabled - routing to research agent")
            return state
        
        # Three-category classification
        prompt = f"""You are a query router. Classify this query into EXACTLY ONE category.

Query: {state.query}

Category "conversational" - Use ONLY for:
- Greetings: hello, hi, hey, good morning, good evening
- Pleasantries: thanks, thank you, bye, goodbye, see you
- Questions about the assistant: who are you, what can you do, how are you
- Pure small talk with NO information request

Category "research" - Use for complex queries requiring deep analysis:
- Requests for comprehensive research or investigation
- Queries asking for "deep dive", "extensive research", "thorough analysis"
- Questions requiring multiple sources or perspectives
- Complex topics needing detailed exploration (e.g., "journey of an AI engineer", "complete guide to...")
- When the query explicitly mentions "research", "investigate", or "analyze deeply"

Category "contextual" - Use for straightforward information requests:
- ANY question asking for facts, definitions, explanations
- Requests like "give me", "tell me about", "explain", "what is", "how does"
- Questions about documents, policies, procedures, workflows
- When in doubt, ALWAYS choose "contextual"

Respond with ONLY the word "conversational", "research", or "contextual":"""

        decision = self.llm.generate(prompt).strip().lower()
        
        if "conversational" in decision and len(decision) < 20:
            state.agent_type = "conversational"
        elif "research" in decision:
            state.agent_type = "research"
        else:
            state.agent_type = "contextual"
        
        state.agent_path.append("router")
        logger.info(f"LLM classified query as: {state.agent_type}")
        return state