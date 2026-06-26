import numpy as np


class EmbeddingUtils:

    @staticmethod
    def validate_dimension(
        embedding
    ):

        if len(embedding) == 768:

            return True

        return False

    @staticmethod
    def cosine_similarity(
        vec1,
        vec2
    ):

        vec1 = np.array(vec1)

        vec2 = np.array(vec2)

        return (
            np.dot(vec1, vec2)
            /
            (
                np.linalg.norm(vec1)
                *
                np.linalg.norm(vec2)
            )
        )