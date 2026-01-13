"""
WindForecast Intelligence Hub - Main orchestrator for multi-agent system
"""
from chatbot.agents.router_agent import RouterAgent
from chatbot.agents.rag_agent import RAGAgent
from chatbot.agents.direct_agent import DirectAgent
from chatbot.agents.web_search_agent import WebSearchAgent
from chatbot.config.config import CHATBOT_NAME, CHATBOT_DESCRIPTION
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WindForecastIntelligenceHub:
    """
    Main orchestrator for the multi-agent AI system
    Routes queries to appropriate specialized agents
    """
    
    def __init__(self):
        self.name = CHATBOT_NAME
        self.description = CHATBOT_DESCRIPTION
        
        # Initialize all agents
        self.router = RouterAgent()
        self.rag_agent = RAGAgent()
        self.direct_agent = DirectAgent()
        self.web_search_agent = WebSearchAgent()
        
        logger.info(f"Initialized {self.name}")
    
    def chat(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a chat query using the multi-agent system
        
        Args:
            query: User's question
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with response, agent_used, and metadata
        """
        try:
            # Route query to appropriate agent
            selected_agent = self.router.process(query)
            logger.info(f"Router selected: {selected_agent} for query: {query[:50]}...")
            
            # Process with selected agent
            if selected_agent == "RAG_AGENT":
                response = self.rag_agent.process(query)
                agent_name = "Report Analysis Agent"
            elif selected_agent == "WEB_SEARCH_AGENT":
                response = self.web_search_agent.process(query)
                agent_name = "Web Research Agent"
            else:
                response = self.direct_agent.process(query)
                agent_name = "Knowledge Agent"
            
            return {
                "response": response,
                "agent_used": agent_name,
                "agent_type": selected_agent,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "agent_used": "Error Handler",
                "agent_type": "ERROR",
                "success": False
            }
    
    def get_welcome_message(self) -> str:
        """Get welcome message for the chatbot"""
        return f"""Welcome to {self.name}! 🌬️

I'm your specialized AI assistant for Wind Energy Forecasting. I can help you with:

📊 **Project Reports** - Ask about data analysis, model training, predictions, or dashboard features
🧠 **Technical Knowledge** - Questions about wind energy, ML models, forecasting techniques
🌐 **Current Information** - Recent developments, news, or up-to-date data

How can I assist you today?"""
