import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.ollama_client import OllamaClient

llm = OllamaClient()

response = llm.generate(
    "What is Artificial Intelligence?"
)

print(response)