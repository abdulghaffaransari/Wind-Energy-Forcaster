"""
Configuration for WindForecast Intelligence Hub
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).parent.parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    # Explicitly specify encoding to avoid UTF-8 decode errors
    load_dotenv(env_path, encoding='utf-8')
else:
    # Try loading from current directory
    load_dotenv(encoding='utf-8')

# Base paths
REPORTS_DIR = BASE_DIR / "Reports"
CHATBOT_DIR = BASE_DIR / "chatbot"

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

# Groq models: llama-3.1-70b-versatile, llama-3.1-8b-instant, llama-3.3-70b-versatile
# Using llama-3.1-70b-versatile - most capable and versatile model
GROQ_MODEL = "llama-3.1-8b-instant"

# Agent Configuration
AGENT_CONFIG = {
    "router": {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "max_tokens": 150
    },
    "rag": {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "max_tokens": 2000,
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "direct": {
        "model": GROQ_MODEL,
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "web_search": {
        "model": GROQ_MODEL,
        "temperature": 0.5,
        "max_tokens": 2000
    }
}

# System Prompts
SYSTEM_PROMPTS = {
    "router": """You are a routing agent for WindForecast Intelligence Hub, a specialized AI assistant for Wind Energy Forecasting.
Your job is to analyze user queries and route them to the appropriate specialized agent:
1. RAG_AGENT - For questions about project reports, data analysis, model training results, predictions, or dashboard features
2. DIRECT_AGENT - For general questions about wind energy, forecasting concepts, or technical explanations
3. WEB_SEARCH_AGENT - For questions requiring current/recent information, news, or data not in the knowledge base

Respond with ONLY one word: RAG_AGENT, DIRECT_AGENT, or WEB_SEARCH_AGENT""",
    
    "rag": """You are a specialized RAG (Retrieval-Augmented Generation) agent for WindForecast Intelligence Hub.
You have access to comprehensive project reports including:
- Data Overview Reports
- Data Analysis Reports  
- Model Training Reports
- Prediction Reports
- Dashboard Reports

Use the provided context from these reports to answer questions accurately. If the context doesn't contain the answer, say so clearly.""",
    
    "direct": """You are a knowledgeable assistant for WindForecast Intelligence Hub, specializing in Wind Energy Forecasting.
You can answer questions about:
- Wind energy concepts and principles
- Machine learning models (LSTM, Transformer, XGBoost, LightGBM, Prophet)
- Time series forecasting techniques
- Feature engineering
- Model evaluation metrics
- General technical questions

Provide clear, accurate, and helpful responses.""",
    
    "web_search": """You are a web search agent for WindForecast Intelligence Hub.
You search the internet for current information, recent developments, news, or data that may not be in the knowledge base.
Use the search results to provide accurate, up-to-date information to users."""
}

# Chatbot Identity
CHATBOT_NAME = "WindForecast Intelligence Hub"
CHATBOT_DESCRIPTION = "Your specialized AI assistant for Wind Energy Forecasting insights and analysis"
