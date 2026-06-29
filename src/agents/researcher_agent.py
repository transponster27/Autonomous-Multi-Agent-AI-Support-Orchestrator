from src.agents.state import AgentState
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder
from src.llm.ollama_client import OllamaClient
from src.utils.logger import logger

class ResearcherAgent:
    """
    Handles complex queries by decomposing them into sub-queries,
    retrieving for each, and synthesizing the combined results.
    """
    
    def __init__(self, retriever: RetrievalPipeline, llm: OllamaClient):
        self.retriever = retriever
        self.llm = llm
    
    def run(self, state: AgentState) -> AgentState:
        """Execute multi-step research workflow."""
        
        logger.info(f"ResearcherAgent processing query: {state.query}")
        
        # Step 1: Decompose query into sub-queries
        sub_queries = self._decompose_query(state.query)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        
        # Step 2: Retrieve for each sub-query
        all_chunks = []
        seen_chunk_ids = set()
        
        for sub_query in sub_queries:
            chunks = self.retriever.retrieve(sub_query, top_k=3)
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
        
        # Limit total chunks to avoid context overflow
        state.retrieved_chunks = all_chunks[:8]
        
        if not state.retrieved_chunks:
            state.final_answer = "No relevant information was found for this research query."
            state.agent_path.append("researcher_agent")
            return state
        
        # Track documents searched
        state.documents_searched = list(set([
            chunk.get("document_name", "Unknown") for chunk in state.retrieved_chunks
        ]))
        
        # Step 3: Synthesize across all retrieved chunks
        context = ContextBuilder.build(state.retrieved_chunks)
        
        synthesis_prompt = f"""You are a senior research analyst. 
You have been given multiple sub-topics to research and a collection of source documents.

Original Research Question: {state.query}

Sub-questions investigated:
{chr(10).join(f'- {sq}' for sq in sub_queries)}

Collected Evidence:
{context}

Instructions:
1. Synthesize a comprehensive answer that addresses the original research question
2. Draw connections between different pieces of evidence
3. If sources contradict each other, explicitly note the contradiction
4. Structure your response with clear sections or bullet points
5. Cite specific documents when making claims

Research Synthesis:"""

        answer = self.llm.generate(synthesis_prompt)
        citations = CitationBuilder.build(state.retrieved_chunks)
        
        state.final_answer = answer
        state.citations = citations
        state.intermediate_reasoning = f"Sub-queries: {sub_queries}"
        state.agent_path.append("researcher_agent")
        
        return state
    
    def _decompose_query(self, query: str) -> list:
        """Use LLM to break complex query into sub-queries."""
        
        decomposition_prompt = f"""Break this complex research question into 2-4 simpler sub-questions 
that, when answered together, would fully address the original question.

Original Question: {query}

Respond with ONLY a JSON array of strings, e.g.:
["sub-question 1", "sub-question 2"]

Sub-questions:"""

        try:
            raw = self.llm.generate(decomposition_prompt)
            
            import json
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            sub_queries = json.loads(text)
            
            if not isinstance(sub_queries, list) or len(sub_queries) == 0:
                return [query]
            
            return sub_queries[:4]  # Cap at 4 sub-queries
            
        except (json.JSONDecodeError, KeyError):
            logger.warning("Query decomposition failed, using original query")
            return [query]