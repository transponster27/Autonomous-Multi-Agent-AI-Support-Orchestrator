from src.agents.state import AgentState
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder

class QAAgent:
    """Handles factual questions about single documents"""
    
    def __init__(self, retriever: RetrievalPipeline, llm):
        self.retriever = retriever
        self.llm = llm
    
    def run(self, state: AgentState) -> AgentState:
        """Retrieve context and generate factual answer"""
        
        # Retrieve relevant chunks
        state.retrieved_chunks = self.retriever.retrieve(state.query, top_k=3)
        
        if not state.retrieved_chunks:
            state.final_answer = "Information not found in the provided documents."
            state.agent_path.append("qa_agent")
            return state
        
        # Build context with source information
        context = ContextBuilder.build(state.retrieved_chunks)
        
        # Build strict factual prompt
        prompt = f"""Answer this factual question using ONLY the provided context.

Rules:
1. Base your answer strictly on the context
2. Do not use outside knowledge
3. If the answer is not in the context, say "Information not found"
4. Be concise and direct

Context:
{context}

Question: {state.query}

Answer:"""

        # Generate answer
        answer = self.llm.generate(prompt)
        
        # Build citations
        citations = CitationBuilder.build(state.retrieved_chunks)
        
        # Update state
        state.final_answer = answer
        state.citations = citations
        state.agent_path.append("qa_agent")
        
        return state