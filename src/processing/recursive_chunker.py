from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class RecursiveChunker:

    def __init__(
        self,
        chunk_size=500,
        overlap=100
    ):

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        )

    def split(
        self,
        text
    ):

        return (
            self.splitter
            .split_text(text)
        )