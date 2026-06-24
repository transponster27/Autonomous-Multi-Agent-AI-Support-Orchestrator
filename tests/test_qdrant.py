# tests/test_qdrant.py

from src.vectorstore.qdrant_manager import (
    QdrantManager
)

db = QdrantManager()

db.create_collection()

print("Qdrant OK")