"""
Web Search Agent - Searches the internet for current information
"""
from chatbot.agents.base_agent import BaseAgent
from typing import Dict, Any, Optional
import requests
from urllib.parse import quote
import logging

logger = logging.getLogger(__name__)


class WebSearchAgent(BaseAgent):
    """Agent that searches the web for current information"""
    
    def __init__(self):
        super().__init__("web_search")
        # Using DuckDuckGo for web search (no API key needed)
        self.search_url = "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1"
    
    def _search_web(self, query: str) -> str:
        """Perform web search and extract relevant information"""
        try:
            # Use DuckDuckGo Instant Answer API
            search_query = quote(query)
            url = self.search_url.format(search_query)
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Extract abstract if available
                if data.get("AbstractText"):
                    return data["AbstractText"]
                elif data.get("Answer"):
                    return data["Answer"]
                elif data.get("RelatedTopics"):
                    # Get first related topic
                    first_topic = data["RelatedTopics"][0] if data["RelatedTopics"] else {}
                    return first_topic.get("Text", "")
            
            # Fallback: Use OpenAI to search (it has web browsing capability in some models)
            return None
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return None
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process query with web search"""
        # Try to get web search results
        web_results = self._search_web(query)
        
        if web_results:
            # Use web results in the prompt
            prompt = f"""Based on the following web search results, answer the user's question.

Web Search Results:
{web_results}

User Question: {query}

Provide a comprehensive answer based on the search results."""
        else:
            # Fallback to direct answer if web search fails
            prompt = f"""The user is asking about current/recent information. Please provide the best answer you can based on your knowledge. If you don't have current information, say so.

User Question: {query}"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages)
        return response
