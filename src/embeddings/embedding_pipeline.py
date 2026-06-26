# src/embeddings/embedding_pipeline.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingPipeline:
    # FIX 1: Accept model_name and load the SentenceTransformer ONCE
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.embedder = SentenceTransformer(model_name, device='cpu')

    # FIX 2: Add the method that routes.py is trying to call
    def generate_embeddings(self, texts):
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        return np.array(embeddings).astype("float32")

    def process_chunks(self, chunks):
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.generate_embeddings(texts)
        return embeddings, texts