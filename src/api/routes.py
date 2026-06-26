# routes.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import APIRouter, UploadFile, File
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.faiss_manager import FaissManager
from src.metadata.metadata import MetadataExtractor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.reranker import Reranker
from src.processing.chunk_manager import ChunkManager
from src.loaders.document_loader import DocumentLoader
from src.processing.cleaner import TextCleaner
from src.generation.generation_pipeline import GenerationPipeline 


router = APIRouter()


# 1. LOAD HEAVY RESOURCES ONCE AT STARTUP

embedder = EmbeddingPipeline(model_name="BAAI/bge-base-en-v1.5")
reranker = Reranker()
faiss_manager = FaissManager(embedding_dim=768)

# FIX: Pass the underlying SentenceTransformer model to ChunkManager.
# This prevents SemanticChunker from loading the model a second time!
chunker = ChunkManager(embedder_model=embedder.embedder) 

loader = DocumentLoader() # Now safely loads PaddleOCR once at startup
generator = GenerationPipeline() 

retriever_pipeline = None


# 2. UPLOAD ENDPOINT

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global retriever_pipeline

    content = await file.read()
    document = loader.load_bytes(content, file.filename)
    
    
    # PAGE-BY-PAGE CHUNKING!
    
    # If it's a PDF, 'pages' exists. If it's TXT/DOCX, we fallback to a single page.
    pages = document.get("pages", [{"page": 1, "text": document["text"]}])
    
    metadata_list = []
    enriched_chunks = []
    chunk_idx = 0
    
    for page_data in pages:
        # Clean the text for this specific page
        page_text = TextCleaner.clean(page_data["text"])
        page_num = page_data["page"]
        
        # Chunk this specific page
        page_raw_chunks = chunker.create_chunks(page_text)
        
        for c in page_raw_chunks:
            chunk_text = c["chunk_text"]
            
            # Create rich metadata with the EXACT page number!
            meta = MetadataExtractor.create(
                filename=file.filename,
                chunk=chunk_text,
                chunk_id=f"chunk_{chunk_idx}_{file.filename}",
                page_number=page_num, # ACCURATE PAGE NUMBER!
                category="general"
            )
            metadata_list.append(meta)
            #merge the metadata into the chunk dictionary
            # Now BM25 will have access to document_name, page_number, etc.
            chunk_dict = {"chunk_text": chunk_text}
            chunk_dict.update(meta) 
            enriched_chunks.append(chunk_dict)
            
            chunk_idx += 1

    if not metadata_list:
         return {"status": "warning", "message": "No text extracted from document."}

    # Generate embeddings for all chunks
    texts = [c["chunk_text"] for c in enriched_chunks]
    embeddings = embedder.generate_embeddings(texts)

    # Save to FAISS
    faiss_manager.add_documents(embeddings, metadata_list)

    # Initialize Pipeline
    retriever_pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)

    return {
        "status": "success",
        "chunks_indexed": len(metadata_list),
        "total_vectors_in_db": faiss_manager.count()
    }

@router.get("/query")
def query(q: str):
    if not retriever_pipeline:
        return {"error": "No documents indexed yet"}
    
    # Step 1: Retrieve the best chunks using Vector + BM25 + Reranker
    docs = retriever_pipeline.retrieve(q)
    
    if not docs:
        return {
            "query": q,
            "answer": "Information not found in the provided documents.",
            "citations": [],
            "sources": []
        }

    # Step 2: Pass the retrieved chunks to the LLM Generator
    generation_result = generator.generate_answer(q, docs)

    # Step 3: Return the final unified response
    return {
        "query": q,
        "answer": generation_result["answer"],
        "citations": generation_result["citations"],
        "sources": docs  # Optional: returns the raw chunks if your frontend needs to highlight text
    }