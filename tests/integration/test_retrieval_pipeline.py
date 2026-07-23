"""Integration tests for the retrieval pipeline."""

import pytest
from src.processing.recursive_chunker import RecursiveChunker
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.faiss_manager import FaissManager
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.reranker import Reranker


class TestRetrievalPipelineIntegration:
    """Integration tests for the full retrieval pipeline."""
    
    @pytest.fixture
    def embedder(self):
        """Load the embedding model."""
        return EmbeddingPipeline(model_name="BAAI/bge-base-en-v1.5")
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks from test text."""
        text = """
        Business Policy Overview
        
        A business policy is a set of guidelines that help an organization achieve its goals.
        Effective policies should be clear, specific, and consistently applied.
        
        Leave Policy
        
        Employees are entitled to 15 days of paid annual leave per year.
        Unused leave may be carried forward up to a maximum of 5 days.
        Sick leave is provided separately at 10 days per year.
        
        Code of Conduct
        
        All employees must adhere to the company's code of conduct.
        Violations may result in disciplinary action up to and including termination.
        """
        
        chunker = RecursiveChunker(chunk_size=200, overlap=50)
        chunks = chunker.split(text)
        
        for i, chunk_text in enumerate(chunks):
            chunks[i] = {
                "chunk_id": f"chunk_{i}",
                "chunk_text": chunk_text,
                "document_name": "test_policy.pdf",
                "page_number": 1
            }
        
        return chunks
    
    @pytest.fixture
    def faiss_manager(self, embedder, sample_chunks):
        """Build a FAISS index from sample chunks."""
        manager = FaissManager(embedding_dim=768)
        texts = [c["chunk_text"] for c in sample_chunks]
        embeddings = embedder.generate_embeddings(texts)
        manager.add_documents(embeddings, sample_chunks)
        return manager
    
    @pytest.fixture
    def reranker(self):
        """Load the reranker model."""
        return Reranker()
    
    @pytest.fixture
    def pipeline(self, sample_chunks, faiss_manager, embedder, reranker):
        """Create the full retrieval pipeline."""
        return RetrievalPipeline(
            chunks=sample_chunks,
            faiss_manager=faiss_manager,
            embedder=embedder,
            reranker=reranker
        )
    
    def test_retrieval_returns_results(self, pipeline):
        """Pipeline should return results for a relevant query."""
        results = pipeline.retrieve("What is the leave policy?")
        assert len(results) > 0
        assert len(results) <= 5
    
    def test_retrieval_returns_relevant_chunks(self, pipeline):
        """Top result should be relevant to the query."""
        results = pipeline.retrieve("How many days of annual leave?")
        found_relevant = any("leave" in r.get("chunk_text", "").lower() or "days" in r.get("chunk_text", "").lower() for r in results)
        assert found_relevant, "No relevant chunks found"
    
    def test_retrieval_respects_top_k(self, pipeline):
        """Pipeline should respect the top_k parameter."""
        results = pipeline.retrieve("business policy", top_k=2)
        assert len(results) <= 2
    
    def test_retrieval_includes_metadata(self, pipeline):
        """Results should include full metadata."""
        results = pipeline.retrieve("code of conduct")
        if results:
            result = results[0]
            assert "chunk_id" in result
            assert "document_name" in result
            assert "page_number" in result
            assert "chunk_text" in result
    
    def test_retrieval_handles_irrelevant_query(self, pipeline):
        """Pipeline should handle queries with no relevant content."""
        results = pipeline.retrieve("What is the weather forecast?")
        assert len(results) > 0