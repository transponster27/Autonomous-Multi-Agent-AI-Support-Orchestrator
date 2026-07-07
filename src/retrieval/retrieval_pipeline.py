from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from typing import List, Optional

class RetrievalPipeline:
    def __init__(self, chunks, faiss_manager, embedder, reranker):
        self.hybrid = HybridRetriever(chunks, faiss_manager, embedder)
        self.reranker = reranker

    def retrieve(
        self, 
        query, 
        top_k=5, 
        document_filter: Optional[List[str]] = None,
        domain_filter: Optional[str] = None  # ✅ Add domain filter
    ):
        # ✅ Pass both filters to hybrid retriever
        candidates = self.hybrid.retrieve(
            query, 
            top_k=10, 
            document_filter=document_filter,
            domain_filter=domain_filter
        )
        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, top_k=top_k)