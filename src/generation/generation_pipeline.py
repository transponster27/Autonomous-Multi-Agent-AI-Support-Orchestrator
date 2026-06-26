# src/generation/generation_pipeline.py

from src.llm.ollama_client import OllamaClient
from src.generation.context_builder import ContextBuilder
from src.generation.prompt_builder import PromptBuilder
from src.generation.citation_builder import CitationBuilder
from src.generation.response_validator import ResponseValidator # Import your validator

class GenerationPipeline:

    def __init__(self):
        self.llm = OllamaClient()

    def generate_answer(self, question, retrieved_docs):
        # Handle edge case where retriever finds nothing
        if not retrieved_docs:
            return {
                "answer": "Information not found in the provided documents.",
                "citations": []
            }

        context = ContextBuilder.build(retrieved_docs)
        prompt = PromptBuilder.build(question, context)
        answer = self.llm.generate(prompt)
        
        # Actually use the validator you wrote!
        if not ResponseValidator.validate(answer):
            answer = "Information not found in the provided documents."

        citations = CitationBuilder.build(retrieved_docs)

        return {
            "answer": answer,
            "citations": citations
        }