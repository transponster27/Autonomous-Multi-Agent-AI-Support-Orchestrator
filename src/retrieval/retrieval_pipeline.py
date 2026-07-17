from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.mmr import MMRSelector
from src.retrieval.reranker import Reranker
from typing import List, Optional

class RetrievalPipeline:
    def __init__(self, chunks, faiss_manager, embedder, reranker, use_mmr=True):
        self.hybrid = HybridRetriever(chunks, faiss_manager, embedder)
        self.reranker = reranker
        self.mmr = MMRSelector(lambda_param=0.5) if use_mmr else None
        self.embedder = embedder

    def retrieve(
        self, 
        query, 
        top_k=5, 
        document_filter: Optional[List[str]] = None,
        domain_filter: Optional[str] = None  #  Add domain filter
    ):
        # Pass both filters to hybrid retriever
        candidates = self.hybrid.retrieve(
            query, 
            top_k=15, 
            document_filter=document_filter,
            domain_filter=domain_filter
        )
        if not candidates:
            return []

        # Rerank first
        reranked = self.reranker.rerank(query, candidates, top_k=10)
        
        # Apply MMR for diversity
        if self.mmr and len(reranked) > top_k:
            # Generate embeddings for reranked chunks
            chunk_texts = [c["chunk_text"] for c in reranked]
            chunk_embeddings = self.embedder.generate_embeddings(chunk_texts)
            query_embedding = self.embedder.generate_embeddings([query])[0]
            
            # Select diverse chunks
            diverse_chunks = self.mmr.select(
                query_embedding,
                chunk_embeddings,
                reranked,
                top_k=top_k
            )
            print(f"MMR selected {len(diverse_chunks)} diverse chunks from {len(reranked)} candidates")
            return diverse_chunks
        
        return reranked[:top_k]