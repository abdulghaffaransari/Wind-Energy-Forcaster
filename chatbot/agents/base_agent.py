"""
Base Agent Class for all specialized agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from groq import Groq
from chatbot.config.config import GROQ_API_KEY, AGENT_CONFIG, SYSTEM_PROMPTS


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.config = AGENT_CONFIG.get(agent_type, {})
        self.system_prompt = SYSTEM_PROMPTS.get(agent_type, "")
        
        # Initialize Groq client
        self.client = Groq(api_key=GROQ_API_KEY)
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a query and return a response"""
        pass
    
    def _call_llm(self, messages: list, **kwargs) -> str:
        """Call Groq API with messages"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.get("model", "llama-3.1-8b-instant"),
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 2000),
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
