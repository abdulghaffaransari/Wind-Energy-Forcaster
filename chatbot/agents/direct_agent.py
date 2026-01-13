"""
Direct Answer Agent - Uses OpenAI's knowledge directly
"""
from chatbot.agents.base_agent import BaseAgent
from typing import Dict, Any, Optional


class DirectAgent(BaseAgent):
    """Agent that provides direct answers using OpenAI's knowledge"""
    
    def __init__(self):
        super().__init__("direct")
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process query with direct knowledge"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self._call_llm(messages)
        return response
