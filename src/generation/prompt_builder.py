from src.generation.prompt_templates import (
    RAG_TEMPLATE
)


class PromptBuilder:

    @staticmethod
    def build(
        question,
        context
    ):

        return RAG_TEMPLATE.format(

            question=question,

            context=context
        )