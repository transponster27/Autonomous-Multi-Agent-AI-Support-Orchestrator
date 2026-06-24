import fitz

from src.utils.logger import logger


class PDFLoader:

    def load(self, filepath):

        try:

            document = fitz.open(filepath)

            text = ""

            pages = []

            for page_num in range(len(document)):

                page = document[page_num]

                page_text = page.get_text()

                pages.append(
                    {
                        "page": page_num + 1,
                        "text": page_text
                    }
                )

                text += page_text + "\n"

            logger.info(
                f"PDF loaded successfully: {filepath}"
            )

            return {
                "text": text,
                "pages": pages,
                "file_type": "pdf"
            }

        except Exception as e:

            logger.error(str(e))

            raise