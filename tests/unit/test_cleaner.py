"""Basic unit tests for the TextCleaner component."""
import pytest
import sys
import os

# Ensure src is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.processing.cleaner import TextCleaner

class TestTextCleaner:
    """Test suite for TextCleaner"""
    
    def test_empty_string_returns_empty(self):
        """Empty input should return empty output."""
        assert TextCleaner.clean("") == ""
    
    def test_none_returns_empty(self):
        """None input should return empty string."""
        assert TextCleaner.clean(None) == ""
    
    def test_removes_extra_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        result = TextCleaner.clean("hello    world")
        assert result == "hello world"
    
    def test_preserves_meaningful_text(self):
        """Core content should be preserved."""
        text = "Business policy is important."
        result = TextCleaner.clean(text)
        assert "Business policy" in result