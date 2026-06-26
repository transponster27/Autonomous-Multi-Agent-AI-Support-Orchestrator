# tests/test_qdrant_methods.py

from qdrant_client import QdrantClient

client = QdrantClient(
    path="qdrant_storage"
)

print(dir(client))