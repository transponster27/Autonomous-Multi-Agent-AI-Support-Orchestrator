# src/generation/generation_pipeline.py

from src.llm.ollama_client import OllamaClient
from src.generation.context_builder import ContextBuilder
from src.generation.prompt_builder import PromptBuilder
from src.generation.citation_builder import CitationBuilder
from src.generation.response_validator import ResponseValidator # Import your validator
from src.utils.logger import logger

class GenerationPipeline:
    def __init__(self):
        self.llm = OllamaClient()

    def generate_answer(self, question, retrieved_docs, chat_history=""):
        if not retrieved_docs:
            return {"answer": "Information not found in the provided documents.", "citations": []}

        try:
            context = ContextBuilder.build(retrieved_docs)
            prompt = PromptBuilder.build(question, context, chat_history=chat_history)
            
            logger.debug(f"Sending prompt to LLM (Length: {len(prompt)} chars)")
            answer = self.llm.generate(prompt)
            
            if not ResponseValidator.validate(answer):
                logger.warning("LLM returned an invalid or empty response.")
                answer = "Information not found in the provided documents."

            citations = CitationBuilder.build(retrieved_docs)

            return {"answer": answer, "citations": citations}

        except Exception as e:
            # ERROR HANDLING: If the LLM or Context builder crashes, 
            # we return a safe fallback instead of crashing the FastAPI route.
            logger.error(f"Generation pipeline failed: {str(e)}")
            return {
                "answer": "I encountered an error while generating the answer. Please try again.",
                "citations": CitationBuilder.build(retrieved_docs) # Still return the sources!
            }