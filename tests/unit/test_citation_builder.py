"""Tests for the CitationBuilder component."""

import pytest
from src.generation.citation_builder import CitationBuilder


class TestCitationBuilder:
    """Test suite for CitationBuilder"""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks with metadata for testing."""
        return [
            {
                "chunk_id": "chunk_0_policy.pdf",
                "document_name": "policy.pdf",
                "page_number": 3,
                "chunk_text": "Employees get 15 days of leave."
            },
            {
                "chunk_id": "chunk_1_policy.pdf",
                "document_name": "policy.pdf",
                "page_number": 3,  # Same page as above
                "chunk_text": "Leave can be carried forward."
            },
            {
                "chunk_id": "chunk_0_handbook.pdf",
                "document_name": "handbook.pdf",
                "page_number": 12,
                "chunk_text": "Sick leave is 10 days."
            },
        ]
    
    def test_builds_citations_from_chunks(self, sample_chunks):
        """Should create citations for each unique source."""
        citations = CitationBuilder.build(sample_chunks)
        assert len(citations) > 0
    
    def test_deduplicates_same_page(self, sample_chunks):
        """Two chunks from same doc+page should produce one citation."""
        citations = CitationBuilder.build(sample_chunks)
        
        # Count citations for policy.pdf page 3
        policy_page_3 = [
            c for c in citations 
            if c.get("document") == "policy.pdf" and c.get("page") == 3
        ]
        assert len(policy_page_3) == 1, "Same page should be deduplicated"
    
    def test_includes_document_name(self, sample_chunks):
        """Each citation should include document name."""
        citations = CitationBuilder.build(sample_chunks)
        for citation in citations:
            assert "document" in citation
            assert citation["document"] != ""
    
    def test_includes_page_number(self, sample_chunks):
        """Each citation should include page number."""
        citations = CitationBuilder.build(sample_chunks)
        for citation in citations:
            assert "page" in citation
    
    def test_empty_chunks_returns_empty_list(self):
        """Empty input should return empty citations."""
        citations = CitationBuilder.build([])
        assert citations == []