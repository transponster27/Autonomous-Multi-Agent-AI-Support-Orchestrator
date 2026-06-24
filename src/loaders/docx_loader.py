from docx import Document

from src.utils.logger import logger


class DOCXLoader:

    def load(self, filepath):

        try:

            document = Document(filepath)

            paragraphs = []

            for para in document.paragraphs:

                paragraphs.append(
                    para.text
                )

            text = "\n".join(paragraphs)

            logger.info(
                f"DOCX loaded: {filepath}"
            )

            return {
                "text": text,
                "file_type": "docx"
            }

        except Exception as e:

            logger.error(str(e))

            raise