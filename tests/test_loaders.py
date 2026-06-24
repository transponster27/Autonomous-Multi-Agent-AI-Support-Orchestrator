from src.loaders.document_loader import (
    DocumentLoader
)

loader = DocumentLoader()

result = loader.load(
    "data/raw/Policy-Statement-on-Business-and-Human-Rights.pdf"
)

print("\nFILE TYPE:")
print(result["file_type"])

print("\nTEXT SAMPLE:\n")

print(
    result["text"][:1000]
)