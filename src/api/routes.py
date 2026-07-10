# routes.py
import os
os.environ["FLAGS_new_ir_enabled"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from src.utils.logger import logger
from src.vectorstore.faiss_manager import FaissManager
import faiss
import numpy as np
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
from src.agents.conversational_agent import ConversationalAgent
from src.agents.state import AgentState
from src.llm.ollama_client import OllamaClient
import asyncio
from src.loaders.file_manager import FileManager
from typing import List, Dict, Optional
from src.agents.research_agent import ResearchAgent
from src.vectorstore.page_index import PageIndex

page_index = PageIndex()
file_manager = FileManager()
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

# ==========================================
# DELETE SINGLE DOCUMENT
# ==========================================
@router.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a specific document from the index"""
    global retrieval_pipeline, orchestrator
    
    try:
        # Find chunks belonging to this document
        chunks_to_keep = []
        chunks_removed = 0
        
        for meta in faiss_manager.metadata:
            if meta.get("document_name") == filename:
                chunks_removed += 1
            else:
                chunks_to_keep.append(meta)
        
        if chunks_removed == 0:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        # Update metadata
        faiss_manager.metadata = chunks_to_keep
        
        # Rebuild FAISS index from scratch
        if chunks_to_keep:
            # Create new index
            new_index = faiss.IndexFlatIP(768)
            
            # Re-embed remaining chunks
            texts = [m["chunk_text"] for m in chunks_to_keep]
            embeddings = embedder.generate_embeddings(texts)
            
            # Normalize and add
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            new_index.add(embeddings_array)
            
            # Replace old index
            faiss_manager.index = new_index
            
            # FIX: Use the existing save() method
            faiss_manager.save()
            
            # Rebuild pipelines
            enriched_chunks = [{"chunk_text": m["chunk_text"], **m} for m in chunks_to_keep]
            retrieval_pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)
            orchestrator = AgentOrchestrator(retrieval_pipeline, llm)
        else:
            # No chunks left - reset everything
            faiss_manager.reset()  # FIX: Use existing reset() method
            retrieval_pipeline = None
            orchestrator = None
        
        logger.info(f"Deleted document: {filename} ({chunks_removed} chunks removed)")
        
        return {
            "status": "success",
            "message": f"Deleted '{filename}' ({chunks_removed} chunks removed)",
            "remaining_vectors": faiss_manager.count()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


# ==========================================
# DELETE ALL DOCUMENTS
# ==========================================
@router.delete("/documents")
def delete_all_documents():
    """Delete all documents from the index"""
    global retrieval_pipeline, orchestrator
    
    try:
        # FIX: Use existing reset() method
        faiss_manager.reset()
        
        # Reset pipelines
        retrieval_pipeline = None
        orchestrator = None
        
        logger.info("All documents deleted")
        
        return {
            "status": "success",
            "message": "All documents deleted",
            "remaining_vectors": 0
        }
    
    except Exception as e:
        logger.exception(f"Delete all failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete all documents")

@router.get("/documents")
def list_documents():
    """List all indexed documents"""
    try:
        documents = list(set(meta.get("document_name", "Unknown") for meta in faiss_manager.metadata))
        
        doc_info = []
        for doc_name in documents:
            chunk_count = sum(1 for m in faiss_manager.metadata if m.get("document_name") == doc_name)
            doc_info.append({
                "filename": doc_name,
                "chunks": chunk_count
            })
        
        return {
            "count": len(documents),
            "documents": doc_info
        }
    except Exception as e:
        logger.exception(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")

# 3. UPLOAD ENDPOINT

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), domain: Optional[str] = None):
    global retrieval_pipeline, orchestrator
    
    # Validate file
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    content = await file.read()
    if len(content) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large")
    
    # Save file immediately
    file_info = file_manager.save_file(content, file.filename)
    
    # Check for duplicate using content hash
    if file_manager.check_duplicate(file_info["hash"], faiss_manager.metadata):
        logger.info(f"Duplicate file detected: {file.filename}")
        return {
            "status": "duplicate",
            "message": "File already indexed",
            "filename": file.filename,
            "hash": file_info["hash"][:16]
        }
    
    # FIX: Process synchronously (no background task)
    try:
        result = await process_file_sync(file_info, domain)
        
        return {
            "status": "uploaded",
            "message": "File uploaded and indexed successfully.",
            "filename": file.filename,
            "hash": file_info["hash"][:16],
            "chunks_indexed": result["chunks_indexed"],
            "domain": domain
        }
    
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process file")


async def process_file_sync(file_info: dict, domain: Optional[str] = None) -> dict:
    """Process file synchronously and return result"""
    global retrieval_pipeline, orchestrator
    
    logger.info(f"Processing: {file_info['filename']}")
    
    # Read saved file
    with open(file_info["filepath"], "rb") as f:
        content = f.read()
    
    # Parse document
    document = loader.load_bytes(content, file_info["filename"])
    pages = document.get("pages", [{"page": 1, "text": document["text"]}])

    # Process chunks
    metadata_list = []
    enriched_chunks = []
    chunk_idx = 0
    
    for page_data in pages:
        page_text = TextCleaner.clean(page_data["text"])
        if not page_text.strip():
            continue
        
        for c in chunker.create_chunks(page_text, strategy="semantic"):
            chunk_text = c["chunk_text"]
            meta = MetadataExtractor.create(
                filename=file_info["filename"],
                chunk=chunk_text,
                chunk_id=f"chunk_{chunk_idx}_{file_info['filename']}",
                page_number=page_data["page"],
                category="general",
                file_hash=file_info["hash"],
                domain=domain  #  Add domain to metadata
            )
            #  Register chunk in page index
            page_index.add_chunk(
                chunk_id=meta["chunk_id"],
                document_name=meta["document_name"],
                page_number=meta["page_number"]
            )
            print(f"Chunk {chunk_idx}: {len(chunk_text)} chars")
            metadata_list.append(meta)
            enriched_chunks.append({"chunk_text": chunk_text, **meta})
            chunk_idx += 1
    
    if not metadata_list:
        raise ValueError("No text extracted from file")
    
    # Generate embeddings and store
    texts = [c["chunk_text"] for c in enriched_chunks]
    embeddings = embedder.generate_embeddings(texts)
    faiss_manager.add_documents(embeddings, metadata_list)
    
    # Rebuild pipelines
    retrieval_pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)
    orchestrator = AgentOrchestrator(retrieval_pipeline, llm)
    
    logger.info(f"Processing complete: {len(metadata_list)} chunks indexed")
    
    return {"chunks_indexed": len(metadata_list)}

@router.get("/page/{document_name}/{page_number}")
def get_page_content(document_name: str, page_number: int):
    """Retrieve all chunks from a specific page"""
    chunk_ids = page_index.get_chunks_for_page(document_name, page_number)
    
    if not chunk_ids:
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Retrieve chunks from FAISS
    chunks = []
    for meta in faiss_manager.metadata:
        if meta["chunk_id"] in chunk_ids:
            chunks.append(meta)
    
    return {
        "document": document_name,
        "page": page_number,
        "chunks": chunks,
        "full_text": "\n\n".join([c["chunk_text"] for c in chunks])
    }


# 4. QUERY ENDPOINT (Multi-Agent)

@router.get("/query")
def query(
    q: str = Query(..., min_length=1),
    session_id: str = "default",
    documents: Optional[str] = None,
    domain: Optional[str] = None,
    web_search: bool = Query(False, description="Enable web search")
):
    try:
        document_filter = None
        if documents:
            document_filter = [doc.strip() for doc in documents.split(",")]
        
        logger.info(f"Query received - web_search: {web_search}, domain: {domain}")
        
        # FIX 2: Handle case where orchestrator is None (no documents uploaded)
        if orchestrator:
            state = orchestrator.run(
                q, 
                session_id, 
                document_filter=document_filter,
                domain=domain,
                web_search_enabled=web_search
            )
        else:
            if web_search:
                research_agent = ResearchAgent(None, llm, web_search_enabled=True)
                state = AgentState(
                    query=q,
                    session_id=session_id,
                    web_search_enabled=True
                )
                state = research_agent.run(state)
            else:
                conv_agent = ConversationalAgent(llm)
                state = AgentState(query=q, session_id=session_id)
                state = conv_agent.run(state)
                state.agent_type = "conversational"
        
        # FIX 3: Always return a response
        print(f"DEBUG: citations = {state.citations}")
        print(f"DEBUG: retrieved_chunks count = {len(state.retrieved_chunks)}") 
        return {
            "query": q,
            "answer": state.final_answer or "I cannot answer that.",
            "sources": state.citations, # Now numbered: [{number, document, page, url}, ...]
            "session_id": session_id,
            "agent_type": state.agent_type,
            "agent_path": state.agent_path,
            "web_search_used": web_search,
            "validation_score": state.validation_score,
            "validation_attempts": state.validation_attempts
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")


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