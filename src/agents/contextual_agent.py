# src/agents/contextual_agent.py

from src.agents.state import AgentState
from src.generation.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder
from src.utils.logger import logger
import re


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
        
        #  Build NUMBERED context (not just plain text)
        context = self._build_numbered_context(state.retrieved_chunks)
        
        #  Strict prompt enforcing numbered citations and no preamble
        prompt = f"""Answer the question using ONLY the provided numbered sources.

STRICT RULES:
1. Start your answer DIRECTLY with the information. NO preamble like "Based on the documents..." or "According to..." or "The documents state..."
2. Cite sources using inline whole numbers like [1], [2], [3] immediately after each claim
3. If information is not in the sources, respond ONLY with: "Information not found in the provided documents."
4. Do NOT include any author's full name from the documents
5. Do NOT include any name of the document
6. Do NOT add concluding remarks or summaries unless explicitly asked
7. Keep the answer concise and factual

Sources:
{context}

Question: {state.query}

Answer (start directly with the information):"""
        
        answer = self.llm.generate(prompt)
        
        #  Clean any preamble the LLM might have added anyway
        answer = self._clean_preamble(answer)
        #  Build numbered citation list
        state.citations = self._build_numbered_citations(state.retrieved_chunks)
        state.final_answer = answer
        state.agent_path.append("contextual_agent")
        
        return state
    
    def _build_numbered_context(self, chunks: list) -> str:
        """Format chunks with [1], [2], etc. numbering"""
        context = ""
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            text = chunk.get("chunk_text", "")
            context += f"[{i}] {doc_name} (Page {page}):\n{text}\n\n"
        return context
    
    def _clean_preamble(self, text: str) -> str:
        """Remove common preamble phrases the LLM might still generate"""
        preamble_patterns = [
            r"^(Based on the provided (documents|sources|context|information)[,.]?\s*)",
            r"^(According to the (documents|sources|provided information)[,.]?\s*)",
            r"^(The (documents|sources|provided context) (state|indicate|show|suggest)[,.]?\s*)",
            r"^(In the provided (documents|context|information)[,.]?\s*)",
            r"^(From the (documents|sources)[,.]?\s*)",
            r"^(As per the (documents|sources)[,.]?\s*)",
        ]
        for pattern in preamble_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()
    
    def _build_numbered_citations(self, chunks: list) -> list:
        """Build numbered citation list WITHOUT full author names"""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            
            #  Strip full names from document name (keep just filename)
            doc_name = self._clean_author_name(doc_name)
            
            citations.append({
                "number": i,
                "document": doc_name,
                "page": page,
                "url": chunk.get("url")
            })
        return citations
    
    def _clean_author_name(self, doc_name: str) -> str:
        """Remove full author names from document names"""
        # Remove common name patterns like "John Smith's Resume" -> "Resume"
        # This is a simple heuristic; adjust based on your actual filenames
        name_patterns = [
            r"^[A-Z][a-z]+\s+[A-Z][a-z]+['']s\s+",  # "John Smith's "
            r"^[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?!Inc|Ltd|Corp)",  # "John Smith Document"
        ]
        for pattern in name_patterns:
            doc_name = re.sub(pattern, "", doc_name)
        return doc_name