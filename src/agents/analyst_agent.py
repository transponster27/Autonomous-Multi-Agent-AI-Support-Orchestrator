from src.agents.state import AgentState
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder
from src.utils.logger import logger

class AnalystAgent:
    """
    Handles complex queries requiring comparison, summarization, 
    or synthesis across multiple documents or chunks.
    """
    
    def __init__(self, retriever: RetrievalPipeline, llm):
        self.retriever = retriever
        self.llm = llm
    
    def run(self, state: AgentState) -> AgentState:
        """Execute the analytical retrieval and generation workflow."""
        
        logger.info(f"AnalystAgent processing query: {state.query}")
        
        # 1. Broader Retrieval Strategy
        # We retrieve more chunks than the QA agent to capture broader context
        # across multiple documents or sections.
        state.retrieved_chunks = self.retriever.retrieve(state.query, top_k=6)
        
        if not state.retrieved_chunks:
            state.final_answer = "No relevant information was found to perform this analysis."
            state.agent_path.append("analyst_agent")
            return state
        
        # Track which documents were actually searched
        state.documents_searched = list(set([
            chunk.get("document_name", "Unknown") for chunk in state.retrieved_chunks
        ]))
        
        # 2. Build Context
        # The ContextBuilder already formats chunks with source headers, 
        # which is critical for the LLM to know which document it is reading.
        context = ContextBuilder.build(state.retrieved_chunks)
        
        # 3. Analytical Prompt Engineering
        # Unlike the QA agent, we instruct the LLM to synthesize and structure.
        prompt = f"""You are an expert document analyst. 
Your task is to analyze, compare, or summarize the provided context based on the user's query.

Guidelines:
1. Synthesize information from multiple sources if provided.
2. If the query asks for a comparison, explicitly state the similarities and differences.
3. If the query asks for a summary, provide a structured overview of the main points.
4. Cite the specific document or section you are drawing information from.
5. If the provided context does not contain enough information to answer the query comprehensively, state exactly what information is missing.

Context:
{context}

Query: {state.query}

Analysis:"""

        # 4. Generate Answer
        answer = self.llm.generate(prompt)
        
        # 5. Build Citations
        citations = CitationBuilder.build(state.retrieved_chunks)
        
        # 6. Update State
        state.final_answer = answer
        state.citations = citations
        state.agent_path.append("analyst_agent")
        
        return state