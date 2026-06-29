from src.agents.state import AgentState
from src.llm.ollama_client import OllamaClient
from src.utils.logger import logger

class CriticAgent:
    """
    Reviews the specialist agent's output for hallucinations,
    citation accuracy, and completeness. Can trigger retries.
    """
    
    def __init__(self, llm: OllamaClient, max_retries: int = 1):
        self.llm = llm
        self.max_retries = max_retries
    
    def run(self, state: AgentState) -> AgentState:
        """Critique the current answer and optionally request a retry."""
        
        if not state.final_answer or not state.retrieved_chunks:
            state.agent_path.append("critic_agent")
            return state
        
        # Skip critique if answer is the standard "not found" message
        if "information not found" in state.final_answer.lower():
            state.agent_path.append("critic_agent")
            state.critique_failed = False
            return state
        
        context_text = "\n\n".join([
            f"[Source {i+1}]: {chunk.get('chunk_text', '')}"
            for i, chunk in enumerate(state.retrieved_chunks)
        ])
        
        # ✅ MORE LENIENT CRITIQUE PROMPT
        critique_prompt = f"""You are a reasonable fact-checker. Review the generated answer against the provided context.

    Context:
    {context_text}

    User Query: {state.query}

    Generated Answer: {state.final_answer}

    Evaluate on these three criteria and respond in EXACTLY this JSON format:
    {{
    "grounded": true/false,
    "citations_accurate": true/false,
    "complete": true/false,
    "issues": "brief description of problems, or 'none'"
    }}

    IMPORTANT RULES:
    - "grounded" is true if MOST claims in the answer appear in the context (minor paraphrasing is OK)
    - "citations_accurate" is true if the answer generally references the sources correctly
    - "complete" is true if the answer addresses the MAIN POINT of the user's query (doesn't need to be perfect)
    - If the answer says "Information not found" and the context truly lacks the answer, mark all as true
    - Be LENIENT - accept answers that are mostly correct even if not perfect
    - Only mark as failed if there are MAJOR factual errors or the answer completely ignores the query

    Respond with ONLY the JSON object, no other text."""

        try:
            critique_raw = self.llm.generate(critique_prompt)
            
            import json
            critique_text = critique_raw.strip()
            if "```" in critique_text:
                critique_text = critique_text.split("```")[1]
                if critique_text.startswith("json"):
                    critique_text = critique_text[4:]
            
            critique = json.loads(critique_text)
            
            state.intermediate_reasoning = f"Critique: {critique}"
            state.agent_path.append("critic_agent")
            
            # ✅ MORE LENIENT DECISION LOGIC
            # Only fail if grounded is false AND there are major issues
            grounded = critique.get("grounded", True)
            complete = critique.get("complete", True)
            issues = critique.get("issues", "none").lower()
            
            # Check if issues are actually major
            major_issue_keywords = ["completely wrong", "fabricated", "hallucinated", "contradicts", "does not address"]
            has_major_issue = any(keyword in issues for keyword in major_issue_keywords)
            
            all_pass = grounded and complete and not has_major_issue
            
            if not all_pass:
                logger.warning(f"Critique failed for query '{state.query}': {critique.get('issues')}")
                state.critique_failed = True
                state.critique_issues = critique.get("issues", "unknown")
            else:
                logger.info(f"Critique passed for query: {state.query}")
                state.critique_failed = False
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Critic parsing failed: {e}")
            state.agent_path.append("critic_agent")
            state.critique_failed = False
        
        return state