from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    VectorParams,
    Distance
)

from src.vectorstore.qdrant_schema import (
    VECTOR_SIZE,
    COLLECTION_NAME
)


class QdrantManager:

    def __init__(self):

        self.client = QdrantClient(
            path="qdrant_storage"
        )

    def create_collection(self):

        collections = (
            self.client.get_collections()
        )

        existing = [

            c.name

            for c in collections.collections
        ]

        if COLLECTION_NAME not in existing:

            self.client.create_collection(

                collection_name=
                COLLECTION_NAME,

                vectors_config=
                VectorParams(

                    size=VECTOR_SIZE,

                    distance=
                    Distance.COSINE
                )
            )

            print(
                "Collection Created"
            )