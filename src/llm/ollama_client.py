# src/llm/ollama_client.py
import requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="llama3.2:3b"):
        self.base_url = base_url
        self.model = model 

    def generate(self, prompt):
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096  
                    }
                },
                timeout=120 
            )
            
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"[OLLAMA ERROR] Connection failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[OLLAMA CRASH REASON]: {e.response.text}")
            return "Error: LLM generation failed."