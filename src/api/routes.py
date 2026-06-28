# routes.py
import os
os.environ["FLAGS_new_ir_enabled"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from src.utils.logger import logger

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

logger.info("Initializing global pipeline components...")
embedder = EmbeddingPipeline(model_name="BAAI/bge-base-en-v1.5")
reranker = Reranker()
faiss_manager = FaissManager(embedding_dim=768)
chunker = ChunkManager(embedder_model=embedder.embedder)
loader = DocumentLoader()
generator = GenerationPipeline()

retriever_pipeline = None


# CONVERSATIONAL MEMORY STORE

# In-memory dictionary to store chat history per session.
# Key: session_id (string), Value: list of turns
chat_histories = {}
MAX_TURNS = 3  # Keep the last 3 Q&A pairs to prevent LLM context overflow (OOM)


# VALIDATION CONSTANTS

MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}


# 2. UPLOAD ENDPOINT

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global retriever_pipeline

    # 1. VALIDATE: File Extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected unsupported file: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # 2. VALIDATE: File Size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"Rejected oversized file: {file.filename} ({file_size_mb:.2f}MB)")
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Max size is {MAX_FILE_SIZE_MB}MB."
        )

    # 3. ERROR HANDLING: The Processing Pipeline
    try:
        logger.info(f"Starting ingestion for: {file.filename}")
        
        document = loader.load_bytes(content, file.filename)
        pages = document.get("pages", [{"page": 1, "text": document["text"]}])
        
        metadata_list = []
        enriched_chunks = []
        chunk_idx = 0
        
        for page_data in pages:
            page_text = TextCleaner.clean(page_data["text"])
            if not page_text.strip():
                continue # Skip empty pages
                
            page_num = page_data["page"]
            page_raw_chunks = chunker.create_chunks(page_text)
            
            for c in page_raw_chunks:
                chunk_text = c["chunk_text"]
                meta = MetadataExtractor.create(
                    filename=file.filename, chunk=chunk_text,
                    chunk_id=f"chunk_{chunk_idx}_{file.filename}",
                    page_number=page_num, category="general"
                )
                metadata_list.append(meta)
                
                # Merge metadata into the chunk for BM25
                chunk_dict = {"chunk_text": chunk_text}
                chunk_dict.update(meta)
                enriched_chunks.append(chunk_dict)
                chunk_idx += 1

        if not metadata_list:
            raise ValueError("No readable text could be extracted from this document.")

        texts = [c["chunk_text"] for c in enriched_chunks]
        embeddings = embedder.generate_embeddings(texts)
        faiss_manager.add_documents(embeddings, metadata_list)

        # Initialize the retrieval pipeline with the new chunks
        retriever_pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)
        
        logger.info(f"Successfully indexed {len(metadata_list)} chunks from {file.filename}")
        return {
            "status": "success",
            "chunks_indexed": len(metadata_list),
            "total_vectors_in_db": faiss_manager.count()
        }

    except ValueError as ve:
        logger.error(f"Validation error during upload: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
        
    except Exception as e:
        logger.exception(f"Critical error processing {file.filename}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing the document."
        )



# 3. QUERY ENDPOINT (With Memory & Error Handling)

@router.get("/query")
def query(q: str = Query(..., min_length=3), session_id: str = "default"):
    if not retriever_pipeline:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a file first.")

    try:
        logger.info(f"Processing query: '{q}' for session: {session_id}")
        
        # Step 1: Retrieve the best chunks using Hybrid Search + Reranker
        docs = retriever_pipeline.retrieve(q)
        
        if not docs:
            return {
                "query": q,
                "answer": "Information not found in the provided documents.",
                "citations": [],
                "sources": [],
                "session_id": session_id
            }

        # Step 2: Fetch and format chat history for this session
        history_list = chat_histories.get(session_id, [])
        history_str = ""
        for turn in history_list:
            history_str += f"{turn['role'].capitalize()}: {turn['content']}\n"

        # Step 3: Pass history to the LLM Generator
        generation_result = generator.generate_answer(q, docs, chat_history=history_str)

        # Step 4: Update the chat history for the next turn
        history_list.append({"role": "user", "content": q})
        history_list.append({"role": "assistant", "content": generation_result["answer"]})
        
        # Trim history to prevent context window overflow (OOM)
        if len(history_list) > MAX_TURNS * 2:
            history_list = history_list[-(MAX_TURNS * 2):]
            
        chat_histories[session_id] = history_list

        # Step 5: Return the final unified response
        return {
            "query": q,
            "answer": generation_result["answer"],
            "citations": generation_result["citations"],
            "sources": docs,
            "session_id": session_id
        }

    except Exception as e:
        logger.exception(f"Error during query generation for: '{q}'")
        raise HTTPException(
            status_code=500, 
            detail="Failed to generate an answer. The LLM might be offline."
        )