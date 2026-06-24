from src.retrieval.retriever import (
    Retriever
)

from src.llm.ollama_client import (
    OllamaClient
)


class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()

        self.llm = OllamaClient()

    def ask(
        self,
        question
    ):

        docs = self.retriever.retrieve(
            question
        )

        context = "\n".join(docs)

        prompt = f"""
Answer only from context.

Context:
{context}

Question:
{question}

If answer not present say:
Information not found in provided documents.
"""

        return self.llm.generate(
            prompt
        )