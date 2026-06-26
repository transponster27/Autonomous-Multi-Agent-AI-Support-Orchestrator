# from sentence_transformers import SentenceTransformer

class VectorRetriever:

    def __init__(self, faiss_manager, embedding_pipeline):
        self.faiss = faiss_manager
        self.embedding_pipeline = embedding_pipeline

    def retrieve(self, query, top_k=5):
        vec = self.embedding_pipeline.generate_embeddings([query])[0]
        return self.faiss.search(vec.tolist(), top_k)