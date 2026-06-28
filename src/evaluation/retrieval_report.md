# RAG Retrieval Evaluation Report
**Generated:** 2026-06-26 14:14:01
**Total Questions Evaluated:** 25
**Total Vectors in DB:** 42

## Performance Metrics
| Metric | Score | Description |
|---|---|---|
| **Hit@1 (Top-1)** | `80.00%` | Correct chunk was the #1 result |
| **Hit@3 (Top-3)** | `88.00%` | Correct chunk was in the top 3 results |
| **Hit@5 (Top-5)** | `92.00%` | Correct chunk was in the top 5 results |

## Pipeline Configuration
- **Embedding Model:** BAAI/bge-base-en-v1.5 (768 dim)
- **Reranker:** BAAI/bge-reranker-base (Cross-Encoder)
- **Search Strategy:** Hybrid (FAISS InnerProduct + BM25 Okapi)

## Failed Retrievals (Hit@5 Misses)
*These are the questions where the correct chunk was NOT found in the top 5 results.*

- **Query:** `What are the main features of an effective business policy?`
  - **Expected:** `chunk_4_Business-Policies-and-Management-Strategies.pdf`
  - **Actually Retrieved:** `chunk_3_Business-Policies-and-Management-Strategies.pdf, chunk_12_Business-Policies-and-Management-Strategies.pdf, chunk_2_Business-Policies-and-Management-Strategies.pdf`

- **Query:** `Why is strategic planning important in organizational profit?`
  - **Expected:** `chunk_9_Business-Policies-and-Management-Strategies.pdf`
  - **Actually Retrieved:** `chunk_10_Business-Policies-and-Management-Strategies.pdf, chunk_12_Business-Policies-and-Management-Strategies.pdf, chunk_27_Business-Policies-and-Management-Strategies.pdf`


## Observations & Limitations
1. **Hybrid Search Efficacy:** Combining BM25 with Vector search successfully captures both exact keyword matches (like specific policy names) and semantic meaning.
2. **Reranker Impact:** The Cross-Encoder successfully re-orders the hybrid results, pushing the most highly relevant context to the very top (improving Hit@1).
3. **Chunking Limitations:** If a concept is split across two chunks during the Recursive Chunking phase, the semantic meaning might be diluted, causing retrieval misses.
4. **Future Improvements:** Implement Parent-Child chunking (retrieving small chunks but sending the whole page to the LLM) to preserve broader context.
5. **Multi-Document Routing:** As more documents are added, the system must correctly route queries to the specific document containing the answer, avoiding cross-contamination.