from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class AgentState:
    """Shared state that flows between agents"""
    
    # Input
    query: str
    session_id: str = "default"
    
    # Router output
    query_type: Optional[str] = None  # "qa", "analysis", "research"
    
    # Retrieval output
    retrieved_chunks: List[Dict] = field(default_factory=list)
    
    # Agent outputs
    intermediate_reasoning: Optional[str] = None
    final_answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)
    
    # Metadata
    documents_searched: List[str] = field(default_factory=list)
    agent_path: List[str] = field(default_factory=list)  # Track which agents ran

    # Add to AgentState dataclass for critic agent
    critique_failed: bool = False
    critique_issues: Optional[str] = None
    retry_count: int = 0