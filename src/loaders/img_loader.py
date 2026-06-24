from paddleocr import PaddleOCR

from src.utils.logger import logger


class ImageLoader:

    def __init__(self):

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en"
        )

    def load(self, filepath):

        try:

            result = self.ocr.ocr(
                filepath,
                cls=True
            )

            extracted_text = []

            for page in result:

                for line in page:

                    extracted_text.append(
                        line[1][0]
                    )

            text = "\n".join(
                extracted_text
            )

            logger.info(
                f"Image OCR complete: {filepath}"
            )

            return {
                "text": text,
                "file_type": "image"
            }

        except Exception as e:

            logger.error(str(e))

            raise