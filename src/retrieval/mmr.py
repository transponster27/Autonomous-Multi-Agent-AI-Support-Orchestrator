import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import logger

class MMRSelector:
    """
    Maximal Marginal Relevance selector.
    Selects top-k chunks while maximizing both relevance and diversity.
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
                         0.7 = favor relevance
                         0.3 = favor diversity
                         0.5 = balanced
        """
        self.lambda_param = lambda_param
    
    def select(self, query_embedding, candidate_embeddings, candidate_chunks, top_k=5):
        """
        Select top-k diverse and relevant chunks.
        
        Args:
            query_embedding: Embedding of the query (1D array)
            candidate_embeddings: Embeddings of candidate chunks (2D array)
            candidate_chunks: List of chunk metadata dicts
            top_k: Number of chunks to select
            
        Returns:
            List of selected chunks (diverse and relevant)
        """
        if len(candidate_chunks) <= top_k:
            return candidate_chunks
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding).reshape(1, -1)
        candidate_vecs = np.array(candidate_embeddings)
        
        # Calculate relevance scores (similarity to query)
        relevance_scores = cosine_similarity(query_vec, candidate_vecs)[0]
        
        # Calculate similarity matrix between all candidates
        similarity_matrix = cosine_similarity(candidate_vecs)
        
        selected_indices = []
        candidate_indices = list(range(len(candidate_chunks)))
        
        # Greedily select top_k chunks
        for _ in range(top_k):
            if not candidate_indices:
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in candidate_indices:
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Max similarity to already selected chunks
                if selected_indices:
                    max_sim = max(similarity_matrix[idx][sel_idx] for sel_idx in selected_indices)
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = (self.lambda_param * relevance) - ((1 - self.lambda_param) * max_sim)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
        
        # Return selected chunks
        return [candidate_chunks[idx] for idx in selected_indices]