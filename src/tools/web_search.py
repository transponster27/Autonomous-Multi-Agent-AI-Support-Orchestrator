from tavily import TavilyClient
from src.utils.logger import logger
import os

class WebSearchTool:
    """Web search tool using Tavily API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set. Web search will be disabled.")
            self.client = None
        else:
            self.client = TavilyClient(api_key=self.api_key)
    
    def search(self, query: str, max_results: int = 5) -> list:
        """Search the web and return results"""
        if not self.client:
            logger.warning("Web search client not initialized")
            return []
        
        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False
            )
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "source": "web"
                })
            
            logger.info(f"Web search returned {len(results)} results for: {query}")
            return results
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []