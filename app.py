from src.rag.rag_pipeline import (
    RAGPipeline
)

rag = RAGPipeline()

while True:

    question = input(
        "\nAsk Question: "
    )

    if question.lower() == "exit":
        break

    answer = rag.ask(
        question
    )

    print("\nAnswer:\n")

    print(answer)