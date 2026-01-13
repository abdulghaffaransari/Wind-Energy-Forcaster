"""
Router Agent - Decides which specialized agent should handle a query
"""
from chatbot.agents.base_agent import BaseAgent
from typing import Dict, Any, Optional


class RouterAgent(BaseAgent):
    """Routes queries to appropriate specialized agents"""
    
    def __init__(self):
        super().__init__("router")
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Route query to appropriate agent"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        response = self._call_llm(messages)
        response = response.strip().upper()
        
        # Validate response
        valid_agents = ["RAG_AGENT", "DIRECT_AGENT", "WEB_SEARCH_AGENT"]
        if response in valid_agents:
            return response
        else:
            # Default routing logic
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ["report", "analysis", "model", "prediction", "dashboard", "data"]):
                return "RAG_AGENT"
            elif any(keyword in query_lower for keyword in ["current", "recent", "news", "latest", "today", "now"]):
                return "WEB_SEARCH_AGENT"
            else:
                return "DIRECT_AGENT"
