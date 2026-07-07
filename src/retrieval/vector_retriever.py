from typing import List, Optional

class VectorRetriever:
    def __init__(self, faiss_manager, embedding_pipeline):
        self.faiss = faiss_manager
        self.embedding_pipeline = embedding_pipeline

    def retrieve(
        self, 
        query, 
        top_k=5, 
        document_filter: Optional[List[str]] = None,
        domain_filter: Optional[str] = None  # ✅ Add domain filter
    ):
        vec = self.embedding_pipeline.generate_embeddings([query])[0]
        
        # Retrieve more candidates if filtering
        search_top_k = top_k * 3 if (document_filter or domain_filter) else top_k
        results = self.faiss.search(vec.tolist(), search_top_k)
        
        # ✅ Apply document filter
        if document_filter:
            results = [
                r for r in results 
                if r.get("document_name") in document_filter
            ]
        
        # ✅ Apply domain filter
        if domain_filter:
            results = [
                r for r in results 
                if r.get("domain") == domain_filter
            ]
            
        return results[:top_k]