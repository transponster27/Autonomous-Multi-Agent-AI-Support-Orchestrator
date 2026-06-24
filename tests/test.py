from sentence_transformers import (
    SentenceTransformer
)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

vector = model.encode(
    "hello world"
)

print(len(vector))