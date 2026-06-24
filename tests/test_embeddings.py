from src.loaders.document_loader import (
    DocumentLoader
)

from src.processing.cleaner import (
    TextCleaner
)

from src.processing.chunk_manager import (
    ChunkManager
)

from src.embeddings.embedding_pipeline import (
    EmbeddingPipeline
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

chunk_manager = (
    ChunkManager()
)

chunks = (
    chunk_manager.create_chunks(
        clean_text,
        strategy="recursive"
    )
)

pipeline = (
    EmbeddingPipeline()
)

results = (
    pipeline.process_chunks(
        chunks,
        cache_key="policy_doc"
    )
)

print(
    f"Chunks: {len(results)}"
)

print(
    len(
        results[0]["embedding"]
    )
)

print(
    results[0]["chunk_text"][:200]
)