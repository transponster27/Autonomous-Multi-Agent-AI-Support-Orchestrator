import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline
)

pipeline = RetrievalPipeline(
    chunks
)

results = pipeline.retrieve(
    "What are business responsibilities?"
)

for item in results:

    print(
        item["chunk_text"][:200]
    )