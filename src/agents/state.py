from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class AgentState:
    query: str
    session_id: str = "default"
    document_filter: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    web_search_enabled: bool = False
    
    # Routing
    agent_type: Optional[str] = None  # conversational or contextual
    
    # Retrieval
    retrieved_chunks: List[Dict] = field(default_factory=list)
    
    # Output
    final_answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)
    
    # Metadata
    documents_searched: List[str] = field(default_factory=list)
    agent_path: List[str] = field(default_factory=list)

    # Validation tracking
    validation_score: Optional[int] = None
    validation_attempts: int = 0
    validation_feedback: Optional[str] = None