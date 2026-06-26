from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

class HybridRetriever:

    def __init__(self, chunks, faiss_manager, embedder):
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(faiss_manager, embedder)

    def retrieve(self, query, top_k=5):

        vector_results = self.vector.retrieve(query, top_k)
        bm25_results = self.bm25.retrieve(query, top_k)

        merged = []
        seen = set()

        for r in vector_results + bm25_results:
            text = r["chunk_text"]
            if text not in seen:
                merged.append(r)
                seen.add(text)

        return merged[:top_k]