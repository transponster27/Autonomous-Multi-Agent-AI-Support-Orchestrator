import re

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


class SemanticChunker:

    def __init__(self, embedder_model):
        self.model = embedder_model

    def split(
        self,
        text,
        threshold=0.65
    ):

        if not text:
            return []

        sentences = [

            s.strip()

            for s in re.split(
                r'(?<=[.!?])\s+',
                text
            )

            if s.strip()
        ]

        if len(sentences) <= 1:
            return [text]

        embeddings = (
            self.model.encode(
                sentences,
                normalize_embeddings=True
            )
        )

        chunks = []

        current_chunk = (
            sentences[0]
        )

        for i in range(
            1,
            len(sentences)
        ):

            similarity = (
                cosine_similarity(
                    [embeddings[i - 1]],
                    [embeddings[i]]
                )[0][0]
            )

            if similarity >= threshold:

                current_chunk += (
                    " "
                    + sentences[i]
                )

            else:

                chunks.append(
                    current_chunk
                )

                current_chunk = (
                    sentences[i]
                )

        chunks.append(
            current_chunk
        )

        return chunks