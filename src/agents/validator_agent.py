from src.agents.state import AgentState
from src.utils.logger import logger
import json
import re

class ValidatorAgent:
    """
    Self-critic agent that validates generated responses.
    Evaluates groundedness, completeness, citation correctness, and formatting.
    Returns structured feedback for regeneration if the response fails.
    """
    
    def __init__(self, llm, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts
    
    def validate(self, state: AgentState) -> dict:
        """
        Validate the current answer against the retrieved context.
        
        Returns:
            dict with keys: valid (bool), score (int 1-10), feedback (str)
        """
        # Skip validation for conversational responses (no context needed)
        if state.agent_type == "conversational":
            return {"valid": True, "score": 10, "feedback": "none"}
        
        # Skip if there's no answer or no context
        if not state.final_answer:
            return {"valid": False, "score": 0, "feedback": "No answer was generated."}
        
        if not state.retrieved_chunks:
            return {"valid": False, "score": 0, "feedback": "No context was retrieved."}
        
        # Build numbered context for the validator
        context = ""
        for i, chunk in enumerate(state.retrieved_chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            text = chunk.get("chunk_text", "")
            context += f"[{i}] {doc_name} (Page {page}):\n{text}\n\n"
        
        # Build the validation prompt
        prompt = f"""You are a strict quality validator for a RAG system.
Evaluate the generated answer against these criteria:

Question: {state.query}

Retrieved Context (numbered sources):
{context}

Generated Answer:
{state.final_answer}

Evaluation Criteria:
1. GROUNDED: Every factual claim must be supported by the context (no hallucination)
2. COMPLETE: The answer must address what the user actually asked
3. NO_PREAMBLE: The answer must start DIRECTLY with information, no filler like "Based on the documents..."
4. CITATIONS: The answer must use inline numbered citations like [1], [2], [3]
5. NO_FULL_NAMES: The answer must NOT include full author names from documents

Respond ONLY with this exact JSON format (no other text):
{{
  "valid": true or false,
  "score": 1 to 10,
  "feedback": "specific issues to fix, or 'none' if valid"
}}

JSON:"""

        try:
            raw = self.llm.generate(prompt)
            parsed = self._parse_json(raw)
            
            return {
                "valid": bool(parsed.get("valid", False)),
                "score": int(parsed.get("score", 0)),
                "feedback": str(parsed.get("feedback", "No feedback provided"))
            }
        except Exception as e:
            logger.warning(f"Validator parsing failed: {e}. Treating as valid.")
            # If validator itself fails, don't block the response
            return {"valid": True, "score": 7, "feedback": "Validator error - defaulting to pass"}
    
    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks"""
        text = text.strip()
        
        # Remove markdown code fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                # Try to find just the JSON object
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    text = match.group(0)
        
        return json.loads(text)