from src.processing.recursive_chunker import (
    RecursiveChunker
)

from src.processing.semantic_chunker import (
    SemanticChunker
)


class ChunkManager:

# FIX: Accept the model instance and pass it down
    def __init__(self, embedder_model=None):
        self.recursive = RecursiveChunker()
        # Only initialize semantic if we actually have a model passed in
        self.semantic = SemanticChunker(embedder_model) if embedder_model else None

    def create_chunks(self, text, strategy="recursive"):
        if strategy == "recursive":
            chunks = self.recursive.split(text)
        elif strategy == "semantic":
            if not self.semantic:
                raise ValueError("Semantic strategy requires an embedder_model.")
            chunks = self.semantic.split(text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        output = []

        for idx, chunk in enumerate(
            chunks
        ):

            output.append({

                "chunk_id":
                    f"chunk_{idx}",

                "chunk_text":
                    chunk
            })

        return output