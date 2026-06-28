# reset_db.py
import sys
import os
import json
import datetime

# Add project root to Python Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.vectorstore.faiss_manager import FaissManager

faiss = FaissManager(embedding_dim=768)
faiss.reset()
print("Database wiped clean! Please upload your file again.")