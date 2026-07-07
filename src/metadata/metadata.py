from pathlib import Path
from datetime import datetime


class MetadataExtractor:

    @staticmethod
    def create(
        filename,
        chunk,
        chunk_id,
        page_number=1,
        category="general",
        file_hash=None,
        domain=None
    ):

        return {

            "chunk_id": chunk_id,

            "document_name": Path(
                filename
            ).name,

            "page_number": page_number,

            "source_type":
            Path(filename)
            .suffix
            .replace(".", ""),

            "category": category,

            "chunk_length": len(chunk),

            "chunk_text": chunk,

            "upload_time":
            datetime.now().isoformat(),

            "file_hash": file_hash,

            "domain": domain 
        }