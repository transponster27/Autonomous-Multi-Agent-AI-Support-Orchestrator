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
from src.agents.orchestrator import AgentOrchestrator
from src.llm.ollama_client import OllamaClient

router = APIRouter()


# 1. GLOBAL INITIALIZATION

logger.info("Initializing global pipeline components...")

# Load heavy models once
embedder = EmbeddingPipeline(model_name="BAAI/bge-base-en-v1.5")
reranker = Reranker()
faiss_manager = FaissManager(embedding_dim=768)
chunker = ChunkManager(embedder_model=embedder.embedder)
loader = DocumentLoader()
generator = GenerationPipeline()
llm = OllamaClient()

# Pipeline state
retrieval_pipeline = None
orchestrator = None

# Conversational memory
chat_histories = {}
MAX_TURNS = 3

# Validation constants
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}

# Auto-restore pipeline if FAISS has data
if faiss_manager.count() > 0:
    logger.info(f"Restoring pipeline from {faiss_manager.count()} existing vectors...")
    enriched_chunks = [{"chunk_text": m["chunk_text"], **m} for m in faiss_manager.metadata]
    retrieval_pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)
    orchestrator = AgentOrchestrator(retrieval_pipeline, llm)
    logger.info("Pipeline restored successfully")


# 2. STATUS ENDPOINT

@router.get("/status")
def check_status():
    """Check FAISS index status and document metadata."""
    try:
        vector_count = faiss_manager.count()
        
        if vector_count == 0:
            return {"status": "empty", "vector_count": 0, "documents": []}
        
        documents = list(set(meta.get("document_name", "Unknown") for meta in faiss_manager.metadata))
        doc_counts = {}
        for meta in faiss_manager.metadata:
            doc_name = meta.get("document_name", "Unknown")
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
        
        return {
            "status": "active",
            "vector_count": vector_count,
            "document_count": len(documents),
            "documents": documents,
            "chunks_per_document": doc_counts,
            "embedding_dimension": faiss_manager.embedding_dim
        }
    except Exception as e:
        logger.exception(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to check status")


# 3. UPLOAD ENDPOINT

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global retrieval_pipeline, orchestrator
    
    # Validate file
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    content = await file.read()
    if len(content) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")
    
    try:
        logger.info(f"Processing: {file.filename}")
        
        # Load and parse document
        document = loader.load_bytes(content, file.filename)
        pages = document.get("pages", [{"page": 1, "text": document["text"]}])
        
        # Process chunks
        metadata_list = []
        enriched_chunks = []
        chunk_idx = 0
        
        for page_data in pages:
            page_text = TextCleaner.clean(page_data["text"])
            if not page_text.strip():
                continue
            
            for c in chunker.create_chunks(page_text):
                chunk_text = c["chunk_text"]
                meta = MetadataExtractor.create(
                    filename=file.filename,
                    chunk=chunk_text,
                    chunk_id=f"chunk_{chunk_idx}_{file.filename}",
                    page_number=page_data["page"],
                    category="general"
                )
                metadata_list.append(meta)
                enriched_chunks.append({"chunk_text": chunk_text, **meta})
                chunk_idx += 1
        
        if not metadata_list:
            raise ValueError("No readable text extracted")
        
        # Generate embeddings and store
        texts = [c["chunk_text"] for c in enriched_chunks]
        embeddings = embedder.generate_embeddings(texts)
        faiss_manager.add_documents(embeddings, metadata_list)
        
        # Rebuild pipelines
        retrieval_pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)
        orchestrator = AgentOrchestrator(retrieval_pipeline, llm)
        
        logger.info(f"Indexed {len(metadata_list)} chunks from {file.filename}")
        return {
            "status": "success",
            "chunks_indexed": len(metadata_list),
            "total_vectors": faiss_manager.count()
        }
    
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")


# 4. QUERY ENDPOINT (Multi-Agent)

@router.get("/query")
def query(q: str = Query(..., min_length=3), session_id: str = "default"):
    if not retrieval_pipeline:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    try:
        state = orchestrator.run(q, session_id)
        
        return {
            "query": q,
            "answer": state.final_answer or "Information not found",
            "citations": state.citations,
            "sources": state.retrieved_chunks,
            "session_id": session_id,
            "query_type": state.query_type,
            "agent_path": state.agent_path
        }
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


# 5. LEGACY QUERY ENDPOINT (Single-Agent)

@router.get("/query-legacy")
def query_legacy(q: str = Query(..., min_length=3), session_id: str = "default"):
    """Original single-agent pipeline for comparison."""
    if not retrieval_pipeline:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    try:
        docs = retrieval_pipeline.retrieve(q)
        
        if not docs:
            return {
                "query": q,
                "answer": "Information not found",
                "citations": [],
                "sources": [],
                "pipeline": "single-agent-legacy"
            }
        
        # Build chat history
        history_list = chat_histories.get(session_id, [])
        history_str = "".join(f"{t['role'].capitalize()}: {t['content']}\n" for t in history_list)
        
        # Generate answer
        result = generator.generate_answer(q, docs, chat_history=history_str)
        
        # Update history
        history_list.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": result["answer"]}
        ])
        if len(history_list) > MAX_TURNS * 2:
            history_list = history_list[-(MAX_TURNS * 2):]
        chat_histories[session_id] = history_list
        
        return {
            "query": q,
            "answer": result["answer"],
            "citations": result["citations"],
            "sources": docs,
            "session_id": session_id,
            "pipeline": "single-agent-legacy"
        }
    except Exception as e:
        logger.exception(f"Legacy query failed: {e}")
        raise HTTPException(status_code=500, detail="Legacy pipeline failed")