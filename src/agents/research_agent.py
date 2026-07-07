from src.agents.state import AgentState
from src.generation.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder
from src.tools.web_search import WebSearchTool
from src.utils.logger import logger

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
        
        # 4. Build context from documents
        doc_context = ""
        if doc_chunks:
            doc_context = ContextBuilder.build(doc_chunks)
            state.documents_searched = list(set([
                chunk.get("document_name", "Unknown") for chunk in doc_chunks
            ]))
        
        # 5. Build context from web results
        web_context = ""
        if web_results:
            web_context = "\n\n".join([
                f"--- Web Source: {r['title']} ---\nURL: {r['url']}\n{r['content']}"
                for r in web_results
            ])
        
        # 6. Build combined prompt
        prompt = self._build_prompt(state.query, doc_context, web_context)
        
        # 7. Generate answer
        answer = self.llm.generate(prompt)
        
        # 8. Build citations (combine doc and web)
        citations = CitationBuilder.build(doc_chunks)
        for r in web_results:
            citations.append({
                "document": r["title"],
                "page": "Web",
                "url": r["url"]
            })
        
        state.final_answer = answer
        state.citations = citations
        state.retrieved_chunks = doc_chunks
        state.agent_path.append("research_agent")
        
        return state
    
    def _build_prompt(self, query: str, doc_context: str, web_context: str) -> str:
        """Build prompt based on available sources"""
        
        if doc_context and web_context:
            return f"""Answer the user's question using BOTH the provided documents AND web sources.

Guidelines:
1. Synthesize information from both internal documents and web sources
2. Clearly distinguish between internal document information and web information
3. If sources conflict, note the discrepancy
4. Cite sources appropriately

Internal Documents:
{doc_context}

Web Sources:
{web_context}

Question: {query}

Answer:"""
        
        elif doc_context:
            return f"""Answer the user's question using the provided documents.
If the answer is not in the documents, say "Information not found in the provided documents."

Documents:
{doc_context}

Question: {query}

Answer:"""
        
        else:  # web_context only
            return f"""Answer the user's question using the provided web sources.
If the answer is not in the web sources, say "Information not found."

Web Sources:
{web_context}

Question: {query}

Answer:"""