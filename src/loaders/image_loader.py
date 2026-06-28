# src/loaders/image_loader.py
import pytesseract
from PIL import Image
from src.utils.logger import logger

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ImageLoader:
    def __init__(self):
        pass  # Tesseract doesn't need heavy model loading!

    def load(self, filepath):
        try:
            # Open image with Pillow
            img = Image.open(filepath)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(img, lang="eng")
            
            logger.info(f"Image OCR complete: {filepath}")
            return {"text": text, "file_type": "image"}

        except Exception as e:
            logger.error(f"OCR failed for {filepath}: {str(e)}")
            raise