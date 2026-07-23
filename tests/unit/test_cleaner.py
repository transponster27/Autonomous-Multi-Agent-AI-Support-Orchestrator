"""Tests for the TextCleaner component."""

import pytest
from src.processing.cleaner import TextCleaner


class TestTextCleaner:
    """Test suite for TextCleaner.clean()"""
    
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
    
    def test_removes_newlines(self):
        """Newlines should be normalized."""
        result = TextCleaner.clean("hello\n\n\nworld")
        assert "\n\n\n" not in result
    
    def test_preserves_meaningful_text(self):
        """Core content should be preserved."""
        text = "Business policy is important."
        result = TextCleaner.clean(text)
        assert "Business policy" in result
        assert "important" in result