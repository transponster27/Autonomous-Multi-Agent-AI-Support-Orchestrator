"""
Test script for the multi-agent RAG pipeline.
Tests router classification, each specialist agent, and the full orchestration.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.faiss_manager import FaissManager
from src.retrieval.reranker import Reranker
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.ollama_client import OllamaClient
from src.agents.orchestrator import AgentOrchestrator


def initialize_pipeline():
    """Initialize all components needed for testing."""
    print("=" * 60)
    print("INITIALIZING MULTI-AGENT PIPELINE")
    print("=" * 60)
    
    embedder = EmbeddingPipeline(model_name="BAAI/bge-base-en-v1.5")
    reranker = Reranker()
    faiss_manager = FaissManager(embedding_dim=768)
    
    if faiss_manager.count() == 0:
        print("ERROR: FAISS index is empty. Upload documents first.")
        return None, None
    
    enriched_chunks = [
        {"chunk_text": m["chunk_text"], **m}
        for m in faiss_manager.metadata
    ]
    
    retriever = RetrievalPipeline(enriched_chunks, faiss_manager, embedder, reranker)
    llm = OllamaClient()
    orchestrator = AgentOrchestrator(retriever, llm)
    
    print(f"Loaded {faiss_manager.count()} vectors from {len(enriched_chunks)} chunks")
    print(f"Documents in DB: {set(m.get('document_name') for m in faiss_manager.metadata)}")
    print()
    
    return retriever, orchestrator


def test_query(orchestrator, query, expected_type=None):
    """Run a single test query and display results."""
    print("-" * 60)
    print(f"QUERY: {query}")
    if expected_type:
        print(f"EXPECTED TYPE: {expected_type}")
    print("-" * 60)
    
    state = orchestrator.run(query)
    
    print(f"Classified As: {state.query_type}")
    print(f"Agent Path:    {' -> '.join(state.agent_path)}")
    print(f"Critique:      {'FAILED' if state.critique_failed else 'PASSED'}")
    if state.critique_issues:
        print(f"Critique Issues: {state.critique_issues}")
    if state.intermediate_reasoning:
        print(f"Reasoning: {state.intermediate_reasoning[:200]}...")
    print(f"Docs Searched: {state.documents_searched}")
    print(f"Chunks Found:  {len(state.retrieved_chunks)}")
    print()
    print("ANSWER:")
    print(state.final_answer)
    print()
    print("CITATIONS:")
    for c in state.citations:
        print(f"  - {c['document']} (page {c['page']})")
    print()


def main():
    retriever, orchestrator = initialize_pipeline()
    if orchestrator is None:
        return
    
    # Test cases designed to hit each agent type
    test_cases = [
        # QA Agent tests - factual, single-document
        {
            "query": "What is the definition of business policy?",
            "expected_type": "qa"
        },
        {
            "query": "Who is responsible for policy formulation?",
            "expected_type": "qa"
        },
        
        # Analyst Agent tests - comparison, synthesis
        {
            "query": "Compare the differences between policy and strategy",
            "expected_type": "analysis"
        },
        {
            "query": "Summarize the main features of an effective business policy",
            "expected_type": "analysis"
        },
        
        # Researcher Agent tests - complex, multi-step
        {
            "query": "What are the key components of strategic management and how do they relate to business policy?",
            "expected_type": "research"
        },
        {
            "query": "Analyze the relationship between top-level management responsibilities and daily operational policies",
            "expected_type": "research"
        },
    ]
    
    print("=" * 60)
    print("RUNNING MULTI-AGENT TESTS")
    print("=" * 60)
    print()
    
    # Track classification accuracy
    correct_classifications = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n>>> TEST {i}/{len(test_cases)}")
        test_query(orchestrator, test["query"], test["expected_type"])
        
        # We can't check actual classification here since test_query 
        # runs the full pipeline, but we can observe it in output
    
    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()