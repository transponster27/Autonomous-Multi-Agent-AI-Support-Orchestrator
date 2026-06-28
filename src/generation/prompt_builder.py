from src.generation.prompt_templates import (
    RAG_TEMPLATE
)

class PromptBuilder:
    @staticmethod
    # chat_history parameter
    def build(question, context, chat_history=""):
        return RAG_TEMPLATE.format(
            question=question,
            context=context,
            chat_history=chat_history
        )