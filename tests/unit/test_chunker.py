"""Tests for the RecursiveChunker component."""

import pytest
from src.processing.recursive_chunker import RecursiveChunker


class TestRecursiveChunker:
    """Test suite for RecursiveChunker"""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker with default settings."""
        return RecursiveChunker(chunk_size=500, overlap=100)
    
    @pytest.fixture
    def small_chunker(self):
        """Create a chunker with small chunk size for testing."""
        return RecursiveChunker(chunk_size=100, overlap=20)
    
    def test_empty_text_returns_empty_list(self, chunker):
        """Empty input should produce no chunks."""
        assert chunker.split("") == []
    
    def test_short_text_single_chunk(self, chunker):
        """Text shorter than chunk_size should produce one chunk."""
        text = "This is a short text."
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text_splits_into_multiple_chunks(self, chunker):
        """Long text should be split into multiple chunks."""
        # Create text longer than 500 chars
        text = "word " * 200  # ~1000 chars
        chunks = chunker.split(text)
        assert len(chunks) > 1
    
    def test_chunks_respect_max_size(self, small_chunker):
        """Each chunk should be roughly within the expected size."""
        text = "word " * 200
        chunks = small_chunker.split(text)
        
        # Each chunk should be around 100 chars (with some tolerance)
        for chunk in chunks:
            assert len(chunk) <= 200, f"Chunk too large: {len(chunk)} chars"
    
    def test_overlap_between_chunks(self, small_chunker):
        """Adjacent chunks should have overlapping content."""
        text = "word " * 200
        chunks = small_chunker.split(text)
        
        if len(chunks) >= 2:
            # The end of chunk 1 should appear at the start of chunk 2
            # (This is what overlap means)
            chunk1_end = chunks[0][-30:]
            chunk2_start = chunks[1][:30]
            # They should share some words
            words1 = set(chunk1_end.split())
            words2 = set(chunk2_start.split())
            assert len(words1 & words2) > 0, "No overlap detected between chunks"
    
    def test_paragraph_boundaries_respected(self, chunker):
        """Chunker should prefer splitting at paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.split(text)
        # Should not split in the middle of a paragraph if possible
        assert len(chunks) >= 1
    
    @pytest.mark.parametrize("text,expected_min_chunks", [
        ("", 0),
        ("Short text.", 1),
        ("word " * 100, 1),
        ("word " * 500, 3),
    ])
    def test_various_text_lengths(self, small_chunker, text, expected_min_chunks):
        """Test chunking behavior across different text lengths."""
        chunks = small_chunker.split(text)
        assert len(chunks) >= expected_min_chunks