# src/retrieval/reranker.py

from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base", device='cpu')

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []

        pairs = [(query, doc["chunk_text"]) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1], 
            reverse=True        
        )

        # : Overwrite the old "score" key with the new Reranker score
        return [
            {**doc, "score": float(score)} 
            for doc, score 
            in ranked[:top_k]
        ]