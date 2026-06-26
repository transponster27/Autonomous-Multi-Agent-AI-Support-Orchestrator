# src/loaders/document_loader.py
import os
import tempfile
from pathlib import Path
from src.loaders.pdf_loader import PDFLoader
from src.loaders.docx_loader import DOCXLoader
from src.loaders.txt_loader import TXTLoader
from src.loaders.image_loader import ImageLoader

class DocumentLoader:

    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.docx_loader = DOCXLoader()
        self.txt_loader = TXTLoader()
        self.image_loader = ImageLoader() 

    def load_bytes(self, file_bytes, filename):
        """
        Handles raw bytes from FastAPI UploadFile.
        Saves to a temporary file and routes to the correct loader.
        """
        # 1. Get the file extension to ensure the temp file is recognized correctly
        extension = Path(filename).suffix.lower()
        
        # 2. Save the raw bytes to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_filepath = tmp_file.name
            
        try:
            # 3. Route to the correct loader using your existing load() method!
            result = self.load(tmp_filepath)
            
            # 4. Ensure the original filename is included in the result for metadata
            result["file_name"] = filename 
            return result
            
        finally:
            # 5. Clean up the temporary file to prevent disk space leaks
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)

    def load(self, filepath):
        # (Keep your existing load() method exactly as it is)
        extension = Path(filepath).suffix.lower()

        if extension == ".pdf":
            return self.pdf_loader.load(filepath)
        elif extension == ".docx":
            return self.docx_loader.load(filepath)
        elif extension == ".txt":
            return self.txt_loader.load(filepath)
        elif extension in [".png", ".jpg", ".jpeg"]:
            return self.image_loader.load(filepath)

        raise ValueError(f"Unsupported file type: {extension}")