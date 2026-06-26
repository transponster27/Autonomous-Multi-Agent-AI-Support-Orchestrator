import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

from src.loaders.document_loader import (
    DocumentLoader
)

loader = DocumentLoader()

result = loader.load(
    "data/raw/Policy-Statement-on-Business-and-Human-Rights.pdf"
)

print("\nFILE TYPE:")
print(result["file_type"])

print("\nTEXT SAMPLE:\n")

print(
    result["text"][:1000]
)