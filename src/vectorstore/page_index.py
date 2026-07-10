from collections import defaultdict
import pickle
import os
from src.utils.logger import logger

class PageIndex:
    """Maps (document_name, page_number) to chunk_ids"""
    
    def __init__(self, index_path="storage/page_index.pkl"):
        self.index_path = index_path
        self.page_to_chunks = defaultdict(list)
        self.chunk_to_page = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    # ✅ FIX: Convert back to defaultdict
                    loaded_page_to_chunks = data.get("page_to_chunks", {})
                    self.page_to_chunks = defaultdict(list, loaded_page_to_chunks)
                    self.chunk_to_page = data.get("chunk_to_page", {})
            except Exception as e:
                logger.warning(f"Failed to load page index: {e}. Starting fresh.")
                self.page_to_chunks = defaultdict(list)
                self.chunk_to_page = {}
    
    def _save(self):
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "page_to_chunks": dict(self.page_to_chunks),
                    "chunk_to_page": self.chunk_to_page
                }, f)
        except Exception as e:
            logger.error(f"Failed to save page index: {e}")
    
    def add_chunk(self, chunk_id, document_name, page_number):
        """Register a chunk with its page location"""
        key = (document_name, page_number)
        # ✅ Now this works because page_to_chunks is a defaultdict
        self.page_to_chunks[key].append(chunk_id)
        self.chunk_to_page[chunk_id] = key
        self._save()
    
    def get_chunks_for_page(self, document_name, page_number):
        """Get all chunk_ids for a specific page"""
        key = (document_name, page_number)
        return self.page_to_chunks.get(key, [])
    
    def get_page_for_chunk(self, chunk_id):
        """Get (document_name, page_number) for a chunk"""
        return self.chunk_to_page.get(chunk_id)
    
    def get_all_pages(self, document_name=None):
        """Get all pages, optionally filtered by document"""
        if document_name:
            return [(doc, page) for doc, page in self.page_to_chunks.keys() if doc == document_name]
        return list(self.page_to_chunks.keys())