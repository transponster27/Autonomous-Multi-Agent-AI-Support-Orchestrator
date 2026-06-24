import pickle
import os


class EmbeddingCache:

    CACHE_DIR = "data/cache"

    def __init__(self):

        os.makedirs(
            self.CACHE_DIR,
            exist_ok=True
        )

    def save(
        self,
        filename,
        embeddings
    ):

        filepath = (
            f"{self.CACHE_DIR}/{filename}.pkl"
        )

        with open(
            filepath,
            "wb"
        ) as file:

            pickle.dump(
                embeddings,
                file
            )

    def load(
        self,
        filename
    ):

        filepath = (
            f"{self.CACHE_DIR}/{filename}.pkl"
        )

        if not os.path.exists(
            filepath
        ):

            return None

        with open(
            filepath,
            "rb"
        ) as file:

            return pickle.load(
                file
            )