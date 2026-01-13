# 🤖 WindForecast Intelligence Hub

A sophisticated multi-agent AI system for Wind Energy Forecasting, integrated into the Streamlit dashboard.

## Overview

WindForecast Intelligence Hub is a specialized AI assistant that uses a multi-agent architecture to provide intelligent responses about wind energy forecasting, project reports, and technical knowledge.

## Architecture

The system consists of four specialized agents:

### 1. Router Agent
- **Purpose**: Analyzes incoming queries and routes them to the most appropriate specialized agent
- **Decision Logic**: 
  - Routes to RAG Agent for project-specific questions
  - Routes to Direct Agent for general knowledge questions
  - Routes to Web Search Agent for current/recent information

### 2. RAG Agent (Report Analysis Agent)
- **Purpose**: Extracts and uses information from project reports (PDFs in Reports folder)
- **Capabilities**:
  - Loads all PDF reports from the Reports directory
  - Performs semantic search to find relevant information
  - Provides answers based on actual project data and analysis

### 3. Direct Agent (Knowledge Agent)
- **Purpose**: Answers questions using Groq's LLM knowledge
- **Capabilities**:
  - Explains wind energy concepts
  - Describes ML models (LSTM, Transformer, XGBoost, etc.)
  - Provides technical explanations about forecasting techniques

### 4. Web Search Agent (Web Research Agent)
- **Purpose**: Searches the internet for current information
- **Capabilities**:
  - Finds recent developments and news
  - Provides up-to-date information not in the knowledge base
  - Uses DuckDuckGo API for web searches

## Folder Structure

```
chatbot/
├── __init__.py
├── intelligence_hub.py      # Main orchestrator
├── README.md
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # Base class for all agents
│   ├── router_agent.py      # Routing logic
│   ├── rag_agent.py         # RAG implementation
│   ├── direct_agent.py      # Direct knowledge agent
│   └── web_search_agent.py  # Web search agent
├── config/
│   ├── __init__.py
│   └── config.py           # Configuration and prompts
└── utils/
    ├── __init__.py
    └── pdf_processor.py     # PDF processing for RAG
```

## Usage

### In Dashboard

The chatbot is integrated into the Streamlit dashboard. Simply:
1. Navigate to "🤖 WindForecast Intelligence Hub" in the sidebar
2. Start chatting with the assistant
3. The system automatically routes your questions to the best agent

### Programmatic Usage

```python
from chatbot.intelligence_hub import WindForecastIntelligenceHub

# Initialize the hub
hub = WindForecastIntelligenceHub()

# Ask a question
result = hub.chat("What models were used in this project?")
print(result["response"])
print(f"Answered by: {result['agent_used']}")
```

## Configuration

Configuration is managed in `chatbot/config/config.py`:
- Groq API key (stored in `.env` file for security)
- Model settings (default: llama-3.1-70b-versatile)
- Agent-specific configurations
- System prompts for each agent
- Paths to reports directory

### Environment Setup

Create a `.env` file in the project root with:
```
GROQ_API_KEY=your_groq_api_key_here
```

The `.env` file is already in `.gitignore` to keep your API key secure.

## Dependencies

- `groq>=0.4.0` - Groq API client (fast LLM inference)
- `python-dotenv>=1.0.0` - Environment variable management
- `PyPDF2>=3.0.0` - PDF text extraction
- `requests>=2.31.0` - Web search API calls

## Features

✅ **Multi-Agent Architecture** - Specialized agents for different query types
✅ **RAG Pipeline** - Retrieval from project reports
✅ **Intelligent Routing** - Automatic agent selection
✅ **Web Search Integration** - Current information retrieval
✅ **Streamlit Integration** - Seamless dashboard experience
✅ **Modular Design** - Easy to extend and maintain

## Example Queries

- **Report Questions**: "What were the model training results?", "Show me data analysis insights"
- **Technical Questions**: "How does LSTM work for time series?", "What is feature engineering?"
- **Current Information**: "What are the latest trends in wind energy?", "Recent wind energy news"

## Notes

- The RAG agent loads all PDFs from the `Reports/` directory on initialization
- The router agent uses keyword matching and LLM-based routing
- All agents use GPT-4 Turbo for high-quality responses
- Chat history is maintained in Streamlit session state
