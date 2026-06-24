from src.embeddings.embedding_service import (
    EmbeddingService
)

from src.embeddings.embedding_cache import (
    EmbeddingCache
)


class EmbeddingPipeline:

    def __init__(self):

        self.embedder = (
            EmbeddingService()
        )

        self.cache = (
            EmbeddingCache()
        )

    def process_chunks(
        self,
        chunks,
        cache_key=None
    ):

        if cache_key:

            cached = self.cache.load(
                cache_key
            )

            if cached:

                return cached

        texts = [

            chunk["chunk_text"]

            for chunk in chunks
        ]

        embeddings = (
            self.embedder
            .generate_embeddings(
                texts
            )
        )

        result = []

        for idx, chunk in enumerate(
            chunks
        ):

            result.append({

                **chunk,

                "embedding":
                    embeddings[idx]
                    .tolist()
            })

        if cache_key:

            self.cache.save(
                cache_key,
                result
            )

        return result