# src/retrieval/retrieval_pipeline.py
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

class RetrievalPipeline:
    # Accept the global reranker instance
    def __init__(self, chunks, faiss_manager, embedder, reranker):
        self.hybrid = HybridRetriever(chunks, faiss_manager, embedder)
        self.reranker = reranker # Use the passed instance instead of creating a new one

    def retrieve(self, query, top_k=5):
        candidates = self.hybrid.retrieve(query, top_k=10)
        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, top_k=top_k)