from pathlib import Path

from src.loaders.pdf_loader import PDFLoader
from src.loaders.docx_loader import DOCXLoader
from src.loaders.txt_loader import TXTLoader
from src.loaders.img_loader import ImageLoader


class DocumentLoader:

    def __init__(self):

        self.pdf_loader = PDFLoader()

        self.docx_loader = DOCXLoader()

        self.txt_loader = TXTLoader()

        self.image_loader = ImageLoader()

    def load(self, filepath):

        extension = (
            Path(filepath)
            .suffix
            .lower()
        )

        if extension == ".pdf":

            return self.pdf_loader.load(
                filepath
            )

        elif extension == ".docx":

            return self.docx_loader.load(
                filepath
            )

        elif extension == ".txt":

            return self.txt_loader.load(
                filepath
            )

        elif extension in [
            ".png",
            ".jpg",
            ".jpeg"
        ]:

            return self.image_loader.load(
                filepath
            )

        raise ValueError(
            f"Unsupported file type: {extension}"
        )