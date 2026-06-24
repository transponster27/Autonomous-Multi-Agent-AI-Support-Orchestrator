from src.loaders.document_loader import (
    DocumentLoader
)

from src.processing.cleaner import (
    TextCleaner
)

from src.processing.chunk_manager import (
    ChunkManager
)

loader = DocumentLoader()

document = loader.load(
    "data/raw/Policy-Statement-on-Business-and-Human-Rights.pdf"
)

clean_text = (
    TextCleaner.clean(
        document["text"]
    )
)

manager = ChunkManager()

chunks = (
    manager.create_chunks(
        clean_text,
        strategy="recursive"
    )
)

print(
    f"Chunks Created: {len(chunks)}"
)

print(
    chunks[0]
)