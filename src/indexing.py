from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_cleaner import TextCleaner

from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator

from src.vectorstore.chroma_manager import ChromaManager


PDF_PATH = (
    "data/raw/Policy-Statement-on-Business-and-Human-Rights.pdf"
)


loader = PDFLoader(PDF_PATH)

text = loader.extract_text()

clean_text = TextCleaner.clean(text)

chunker = DocumentChunker()

chunks = chunker.split(clean_text)

embedder = EmbeddingGenerator()

embeddings = embedder.generate(chunks)

vector_db = ChromaManager()

vector_db.add_chunks(
    chunks,
    embeddings
)

print("Indexing Complete")