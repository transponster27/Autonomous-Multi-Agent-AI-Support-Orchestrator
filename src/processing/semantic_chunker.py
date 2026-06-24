from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

import numpy as np


class SemanticChunker:

    def __init__(self):

        self.model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

    def split(
        self,
        text,
        threshold=0.65
    ):

        sentences = (
            text.split(". ")
        )

        if len(sentences) <= 1:

            return [text]

        embeddings = (
            self.model.encode(
                sentences
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
                    [embeddings[i-1]],
                    [embeddings[i]]
                )[0][0]
            )

            if similarity > threshold:

                current_chunk += (
                    ". "
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