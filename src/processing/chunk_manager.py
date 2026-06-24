from src.processing.recursive_chunker import (
    RecursiveChunker
)

from src.processing.semantic_chunker import (
    SemanticChunker
)


class ChunkManager:

    def __init__(self):

        self.recursive = (
            RecursiveChunker()
        )

        self.semantic = (
            SemanticChunker()
        )

    def create_chunks(
        self,
        text,
        strategy="recursive"
    ):

        if strategy == "recursive":

            chunks = (
                self.recursive
                .split(text)
            )

        elif strategy == "semantic":

            chunks = (
                self.semantic
                .split(text)
            )

        else:

            raise ValueError(
                "Unknown strategy"
            )

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

    def insert_chunks(
        self,
        embedded_chunks
    ):

        points = []

        for idx, chunk in enumerate(
            embedded_chunks
        ):

            points.append({

                "id": idx,

                "vector":
                    chunk["embedding"],

                "payload": {

                    "chunk_id":
                        chunk["chunk_id"],

                    "text":
                        chunk["chunk_text"]
                }
            })

        self.client.upsert(
            collection_name=
            COLLECTION_NAME,

            points=points
        )

        return output