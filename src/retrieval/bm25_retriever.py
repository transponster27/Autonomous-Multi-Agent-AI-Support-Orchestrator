# src/retrieval/bm25_retriever.py
from rank_bm25 import BM25Okapi
from typing import List, Optional

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks or []
        if not self.chunks:
            self.bm25 = None
            return

        corpus = [c["chunk_text"].split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, top_k=5, document_filter: Optional[List[str]] = None, domain_filter: Optional[str] = None):
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        #Return the FULL dictionary from self.chunks, plus the score!
        # Previously, this was stripping everything except chunk_text.
        results = [{**self.chunks[i], "score": float(score)}
            for i, score in ranked]
            
        if document_filter:
            results = [
                r for r in results 
                if r.get("document_name") in document_filter
            ]
        if domain_filter:
            results = [
                r for r in results 
                if r.get("domain") == domain_filter
            ]
            
        return results[:top_k]