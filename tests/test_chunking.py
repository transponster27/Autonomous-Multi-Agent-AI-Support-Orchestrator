# tests/test_chunking.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.chunk_manager import (
    ChunkManager
)

text = """

Artificial Intelligence is transforming industries.

Machine learning is a subset of AI.

Deep learning uses neural networks.

Human rights policies guide organizations.

"""

manager = ChunkManager()

chunks = manager.create_chunks(
    text,
    strategy="semantic"
)

print()

for chunk in chunks:

    print(
        chunk["chunk_id"]
    )

    print(
        chunk["chunk_text"]
    )

    print("-" * 50)