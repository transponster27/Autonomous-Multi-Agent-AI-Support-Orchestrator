from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from src.api.routes import router

app = FastAPI(

    title="RAG System",

    version="1.0"

)

@app.get("/")
def root():

    return {

        "message": "RAG API Running"

    }

app.include_router(router)
