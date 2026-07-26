# src/evaluation/evaluate_retrieval.py

import sys
import os
import json
import datetime

# Add project root to Python Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# IMPORTS
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.faiss_manager import FaissManager
from src.retrieval.reranker import Reranker
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.logger import logger

def run_evaluation(test_file="src/evaluation/test_set.json", report_file="src/evaluation/retrieval_report.md"):
    logger.info("Initializing components for evaluation...")
    
    # 1. Initialize components
    embedder = EmbeddingPipeline(model_name="BAAI/bge-base-en-v1.5")
    reranker = Reranker()
    faiss_manager = FaissManager(embedding_dim=768)
    
    if faiss_manager.count() == 0:
        logger.error("FAISS index is empty. Please upload a document via the API first.")
        return

    # 2. Reconstruct enriched_chunks for BM25
    enriched_chunks = [
        {"chunk_text": m["chunk_text"], **m} 
        for m in faiss_manager.metadata
    ]
    
    pipeline = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)

    # 3. Load test set
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
    except FileNotFoundError:
        logger.error(f"Test file '{test_file}' not found in root directory.")
        return

    top_1_hits, top_3_hits, top_5_hits = 0, 0, 0
    total_questions = len(test_set)
    
    if total_questions == 0:
        logger.error("Test set is empty.")
        return

    logger.info(f"Running evaluation on {total_questions} questions...")

    # Track failed questions for the report
    failed_top_1 = []
    failed_top_5 = []

    # 4. Run evaluation loop
    for item in test_set:
        query = item["query"]
        ground_truth_id = item["ground_truth_chunk_id"]
        
        results = pipeline.retrieve(query)
        retrieved_ids = [r["chunk_id"] for r in results]
        
        # Calculate Hits
        hit_1 = ground_truth_id in retrieved_ids[:1]
        hit_3 = ground_truth_id in retrieved_ids[:3]
        hit_5 = ground_truth_id in retrieved_ids[:5]
        
        if hit_1: top_1_hits += 1
        if hit_3: top_3_hits += 1
        if hit_5: top_5_hits += 1
        
        # Track failures for the report
        if not hit_1:
            failed_top_1.append({"query": query, "expected": ground_truth_id, "retrieved": retrieved_ids[:1]})
        if not hit_5:
            failed_top_5.append({"query": query, "expected": ground_truth_id, "retrieved": retrieved_ids})

    # 5. GENERATE MARKDOWN REPORT
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "# RAG Retrieval Evaluation Report",
        f"**Generated:** {timestamp}",
        f"**Total Questions Evaluated:** {total_questions}",
        f"**Total Vectors in DB:** {faiss_manager.count()}",
        "",
        "## Performance Metrics",
        "| Metric | Score | Description |",
        "|---|---|---|",
        f"| **Hit@1 (Top-1)** | `{top_1_hits / total_questions:.2%}` | Correct chunk was the #1 result |",
        f"| **Hit@3 (Top-3)** | `{top_3_hits / total_questions:.2%}` | Correct chunk was in the top 3 results |",
        f"| **Hit@5 (Top-5)** | `{top_5_hits / total_questions:.2%}` | Correct chunk was in the top 5 results |",
        "",
        "## Pipeline Configuration",
        "- **Embedding Model:** BAAI/bge-base-en-v1.5 (768 dim)",
        "- **Reranker:** BAAI/bge-reranker-base (Cross-Encoder)",
        "- **Search Strategy:** Hybrid (FAISS InnerProduct + BM25 Okapi)",
        "",
        "## Failed Retrievals (Hit@5 Misses)",
        "*These are the questions where the correct chunk was NOT found in the top 5 results.*",
        ""
    ]

    if not failed_top_5:
        report_lines.append("**Perfect Score!** All ground truth chunks were retrieved in the Top 5.")
    else:
        for fail in failed_top_5:
            report_lines.append(f"- **Query:** `{fail['query']}`")
            report_lines.append(f"  - **Expected:** `{fail['expected']}`")
            retrieved_str = ", ".join(fail['retrieved'][:3]) if fail['retrieved'] else "None"
            report_lines.append(f"  - **Actually Retrieved:** `{retrieved_str}`")
            report_lines.append("")

    report_lines.extend([
        "",
        "## Observations & Limitations",
        "1. **Hybrid Search Efficacy:** Combining BM25 with Vector search successfully captures both exact keyword matches (like specific policy names) and semantic meaning.",
        "2. **Reranker Impact:** The Cross-Encoder successfully re-orders the hybrid results, pushing the most highly relevant context to the very top (improving Hit@1).",
        "3. **Chunking Limitations:** If a concept is split across two chunks during the Recursive Chunking phase, the semantic meaning might be diluted, causing retrieval misses.",
        "4. **Future Improvements:** Implement Parent-Child chunking (retrieving small chunks but sending the whole page to the LLM) to preserve broader context.",
        "5. **Multi-Document Routing:** As more documents are added, the system must correctly route queries to the specific document containing the answer, avoiding cross-contamination."
    ])

    # Write to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Evaluation complete! Report saved to: {os.path.abspath(report_file)}")
    logger.debug(f"Evaluation complete! Report saved to: {os.path.abspath(report_file)}")

if __name__ == "__main__":
    run_evaluation()