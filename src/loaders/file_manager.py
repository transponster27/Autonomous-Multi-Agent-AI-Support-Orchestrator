import os
import hashlib
import shutil
from pathlib import Path
from src.utils.logger import logger

class FileManager:
    UPLOAD_DIR = "storage/uploads"
    
    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
    
    def compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    def save_file(self, content: bytes, filename: str) -> dict:
        """Save file to disk and return metadata"""
        file_hash = self.compute_hash(content)
        
        # Create unique filename with hash prefix to avoid collisions
        safe_filename = f"{file_hash[:8]}_{filename}"
        filepath = os.path.join(self.UPLOAD_DIR, safe_filename)
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved file: {filename} (hash: {file_hash[:16]})")
        
        return {
            "filepath": filepath,
            "filename": filename,
            "hash": file_hash,
            "size": len(content)
        }
    
    def check_duplicate(self, file_hash: str, existing_metadata: list) -> bool:
        """Check if file with same hash already exists"""
        for meta in existing_metadata:
            if meta.get("file_hash") == file_hash:
                return True
        return False