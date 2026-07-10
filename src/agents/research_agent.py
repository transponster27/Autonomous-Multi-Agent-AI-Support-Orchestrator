from src.agents.state import AgentState
from src.tools.web_search import WebSearchTool
from src.utils.logger import logger
import re 

class ResearchAgent:
    """Handles complex queries with optional web search"""
    
    def __init__(self, retriever, llm, web_search_enabled=False):
        self.retriever = retriever
        self.llm = llm
        self.web_search = WebSearchTool()
        self.web_search_enabled = web_search_enabled
    
    def run(self, state: AgentState) -> AgentState:
        logger.info(f"ResearchAgent processing: {state.query}")
        logger.info(f"Web search enabled: {self.web_search_enabled}")
        
        # 1. Retrieve from documents
        doc_chunks = []
        if self.retriever:
            doc_chunks = self.retriever.retrieve(
                state.query,
                top_k=5,
                document_filter=state.document_filter if state.document_filter else None,
                domain_filter=state.domain
            )
        
        # 2. Web search if enabled
        web_results = []
        if self.web_search_enabled:
            web_results = self.web_search.search(state.query, max_results=3)
        
        # 3. Check if we have any sources
        if not doc_chunks and not web_results:
            state.final_answer = "Information not found in the provided documents or web sources."
            state.agent_path.append("research_agent")
            return state
        
        # Build numbered contexts (both methods now exist)
        doc_context = self._build_numbered_doc_context(doc_chunks)
        web_context = self._build_numbered_web_context(web_results, start_num=len(doc_chunks) + 1)
        
        # Build combined prompt
        prompt = self._build_prompt(state.query, doc_context, web_context)
        answer = self.llm.generate(prompt)
        answer = self._clean_preamble(answer)
        
        # Build unified numbered citations (docs + web)
        citations = self._build_numbered_citations(doc_chunks, web_results)
        
        state.final_answer = answer
        state.citations = citations
        state.retrieved_chunks = doc_chunks
        state.documents_searched = list(set([
            chunk.get("document_name", "Unknown") for chunk in doc_chunks
        ]))
        state.agent_path.append("research_agent")
        
        return state
    
    # Added missing method
    def _build_numbered_doc_context(self, chunks: list) -> str:
        """Format document chunks with [1], [2], etc. numbering"""
        context = ""
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            text = chunk.get("chunk_text", "")
            context += f"[{i}] {doc_name} (Page {page}):\n{text}\n\n"
        return context
    
    def _build_numbered_web_context(self, results: list, start_num: int) -> str:
        """Format web search results with continuing numbering"""
        context = ""
        for i, r in enumerate(results, start_num):
            context += f"[{i}] {r['title']}:\nURL: {r['url']}\n{r['content']}\n\n"
        return context
    
    def _build_prompt(self, query: str, doc_context: str, web_context: str) -> str:
        """Build prompt based on available sources"""
        if doc_context and web_context:
            return f"""Answer using the numbered sources below.

STRICT RULES:
1. Start DIRECTLY with the answer. NO preamble.
2. Use inline citations [1], [2], etc. after each claim.
3. Do NOT include author full names.
4. Distinguish between internal documents and web sources through citations.
5. If information conflicts between sources, note the discrepancy.

Internal Documents:
{doc_context}

Web Sources:
{web_context}

Question: {query}

Answer (start directly):"""
        
        elif doc_context:
            return f"""Answer using ONLY the provided numbered sources.

STRICT RULES:
1. Start DIRECTLY. NO preamble.
2. Use inline [1], [2] citations.
3. Do NOT include author full names.
4. If not in sources, say "Information not found in the provided documents."

Documents:
{doc_context}

Question: {query}

Answer (start directly):"""
        
        else:
            return f"""Answer using the numbered web sources.

STRICT RULES:
1. Start DIRECTLY. NO preamble.
2. Use inline [1], [2] citations.
3. If not in sources, say "Information not found."

Web Sources:
{web_context}

Question: {query}

Answer (start directly):"""
    
    def _clean_preamble(self, text: str) -> str:
        """Remove common preamble phrases"""
        preamble_patterns = [
            r"^(Based on the provided (documents|sources|context|information)[,.]?\s*)",
            r"^(According to the (documents|sources|provided information)[,.]?\s*)",
            r"^(The (documents|sources|provided context) (state|indicate|show|suggest)[,.]?\s*)",
            r"^(In the provided (documents|context|information)[,.]?\s*)",
        ]
        for pattern in preamble_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()
    
    def _build_numbered_citations(self, doc_chunks: list, web_results: list) -> list:
        """Build unified numbered citations for docs and web"""
        citations = []
        
        # Document citations
        for i, chunk in enumerate(doc_chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            citations.append({
                "number": i,
                "document": doc_name,
                "page": page,
                "url": None
            })
        
        # Web citations (continue numbering)
        for i, r in enumerate(web_results, len(doc_chunks) + 1):
            citations.append({
                "number": i,
                "document": r.get("title", "Web Source"),
                "page": "Web",
                "url": r.get("url")
            })
        
        return citations