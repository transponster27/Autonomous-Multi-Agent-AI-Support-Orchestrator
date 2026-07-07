# src/agents/contextual_agent.py

from src.agents.state import AgentState
from src.generation.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder
from src.utils.logger import logger

class ContextualAgent:
    """Handles document-specific queries with retrieval"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"ContextualAgent processing: {state.query}")

        state.retrieved_chunks = self.retriever.retrieve(
            state.query,
            top_k=5,
            document_filter=state.document_filter if state.document_filter else None,
            domain_filter=state.domain  # Filter by domain
        )
        
        if not state.retrieved_chunks:
            state.final_answer = "Information not found in the provided documents."
            state.agent_path.append("contextual_agent")
            return state
        
        state.documents_searched = list(set([
            chunk.get("document_name", "Unknown") for chunk in state.retrieved_chunks
        ]))
        
        context = ContextBuilder.build(state.retrieved_chunks)
        
        # ✅ FIXED PROMPT: More flexible, handles forms and sparse content
        domain_instruction = f"Focus specifically on the {state.domain} domain if applicable. " if state.domain else ""
        
        prompt = f"""{domain_instruction}Answer the user's question based on the provided documents.

Guidelines:
1. If the documents contain relevant information, provide a clear answer
2. If the documents contain form fields, templates, or structured data, describe what information the form collects or what the document contains
3. If the documents are completely unrelated to the question, say "Information not found in the provided documents"
4. Be helpful and extract whatever useful information is available

Context:
{context}

Question: {state.query}

Answer:"""
        
        answer = self.llm.generate(prompt)
        citations = CitationBuilder.build(state.retrieved_chunks)
        
        state.final_answer = answer
        state.citations = citations
        state.agent_path.append("contextual_agent")
        
        return state