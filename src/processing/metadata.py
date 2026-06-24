from pathlib import Path


class MetadataExtractor:

    @staticmethod
    def extract(
        filepath,
        document_data
    ):

        filename = Path(
            filepath
        ).name

        metadata = {
            "chunk_id":"12",

            "document_name":
                "policy.pdf",

            "page_number":
                3,

            "source_type":
                "pdf",

            "category":
                "human_rights",

            "chunk_text":
                "..."
            }

        return metadata