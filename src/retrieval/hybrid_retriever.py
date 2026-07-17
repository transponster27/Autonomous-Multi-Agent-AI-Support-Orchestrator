from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from typing import List, Optional

class HybridRetriever:
    def __init__(self, chunks, faiss_manager, embedder):
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(faiss_manager, embedder)

    def retrieve(
        self, 
        query, 
        top_k=5, 
        document_filter: Optional[List[str]] = None,
        domain_filter: Optional[str] = None  #  Add domain filter
    ):
        #  Pass both filters to underlying retrievers
        vector_results = self.vector.retrieve(
            query, 
            top_k, 
            document_filter=document_filter,
            domain_filter=domain_filter
        )
        bm25_results = self.bm25.retrieve(
            query, 
            top_k, 
            document_filter=document_filter,
            domain_filter=domain_filter
        )

        # Merge and deduplicate
        merged = []
        seen = set()

        for r in vector_results + bm25_results:
            text = r.get("chunk_text", "")
            if text and text not in seen:
                merged.append(r)
                seen.add(text)

        return merged[:top_k]