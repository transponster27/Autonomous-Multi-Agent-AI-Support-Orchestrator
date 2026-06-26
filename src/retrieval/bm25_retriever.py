# src/retrieval/bm25_retriever.py
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks or []
        if not self.chunks:
            self.bm25 = None
            return

        corpus = [c["chunk_text"].split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, top_k=5):
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        # ✅ FIX: Return the FULL dictionary from self.chunks, plus the score!
        # Previously, this was stripping everything except chunk_text.
        return [
            {**self.chunks[i], "score": float(score)}
            for i, score in ranked
        ]