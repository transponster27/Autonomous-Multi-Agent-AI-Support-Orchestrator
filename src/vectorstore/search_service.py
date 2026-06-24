from sentence_transformers import (
    SentenceTransformer
)

from qdrant_client import (
    QdrantClient
)

from src.vectorstore.qdrant_schema import (
    COLLECTION_NAME
)


class SearchService:

    def __init__(self):

        self.embedder = (
            SentenceTransformer(
                "BAAI/bge-small-en-v1.5"
            )
        )

        self.client = (
            QdrantClient(
                path="qdrant_storage"
            )
        )

    def search(
        self,
        query,
        top_k=5
    ):

        query_vector = (
            self.embedder.encode(
                query
            )
            .tolist()
        )

        results = self.client.search(

            collection_name=
            COLLECTION_NAME,

            query_vector=
            query_vector,

            limit=top_k
        )

        return results