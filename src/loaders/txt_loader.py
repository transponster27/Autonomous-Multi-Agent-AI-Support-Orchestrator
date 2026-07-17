# src/loaders/txt_loader.py
from src.utils.logger import logger

class TXTLoader:
    def load(self, filepath):
        try:
            # Try UTF-8 first. If it fails, fallback to Windows-1252.
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    text = file.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 failed for {filepath}. Falling back to cp1252.")
                with open(filepath, "r", encoding="cp1252") as file:
                    text = file.read()

            logger.info(f"TXT loaded: {filepath}")
            return {"text": text, "file_type": "txt"}

        except Exception as e:
            logger.error(str(e))
            raise