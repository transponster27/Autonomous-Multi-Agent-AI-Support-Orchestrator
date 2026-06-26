import os
import pickle

import faiss
import numpy as np


class FaissManager:

    def __init__(
        self,
        embedding_dim=768,
        index_path="storage/faiss.index",
        metadata_path="storage/metadata.pkl"
    ):

        self.embedding_dim = embedding_dim

        self.index_path = index_path

        self.metadata_path = metadata_path

        os.makedirs("storage", exist_ok=True)

        if (
            os.path.exists(index_path)
            and
            os.path.exists(metadata_path)
        ):

            self.index = faiss.read_index(
                index_path
            )

            with open(
                metadata_path,
                "rb"
            ) as f:

                self.metadata = pickle.load(f)

        else:

            self.index = faiss.IndexFlatIP(
                embedding_dim
            )

            self.metadata = []
    
    def add_documents(
        self,
        embeddings,
        metadata
    ):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        faiss.normalize_L2(
            embeddings
        )

        self.index.add(
            embeddings
        )

        self.metadata.extend(
            metadata
        )

        self.save()
        print("Vectors Added:", len(embeddings))
        print("Index Size:", self.index.ntotal)
    
    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query = np.asarray(
            [query_embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(
            query
        )

        scores, indices = self.index.search(
            query,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:

                continue

            results.append({

                "score": float(score),

                **self.metadata[idx]
            })

        return results
    
    def save(self):

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(
            self.metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )
    def count(self):

        return self.index.ntotal
    
    def reset(self):

        self.index = faiss.IndexFlatIP(
            self.embedding_dim
        )

        self.metadata = []

        self.save()
    
    def get_metadata(self):

        return self.metadata