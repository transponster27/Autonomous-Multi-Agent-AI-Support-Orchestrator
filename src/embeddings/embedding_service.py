from sentence_transformers import (
    SentenceTransformer
)

from src.utils.logger import logger


class EmbeddingService:

    def __init__(
        self,
        model_name="BAAI/bge-small-en-v1.5"
    ):

        logger.info(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

    def generate_embedding(
        self,
        text
    ):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def generate_embeddings(
        self,
        texts
    ):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )