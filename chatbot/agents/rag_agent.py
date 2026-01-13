"""
RAG Agent - Retrieval-Augmented Generation using project reports
"""
from chatbot.agents.base_agent import BaseAgent
from chatbot.utils.pdf_processor import PDFProcessor
from chatbot.config.config import REPORTS_DIR
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Agent that uses RAG to answer questions from project reports"""
    
    def __init__(self):
        super().__init__("rag")
        self.pdf_processor = PDFProcessor(REPORTS_DIR)
        self.reports = None
        self._load_reports()
    
    def _load_reports(self):
        """Load all reports into memory"""
        try:
            self.reports = self.pdf_processor.load_all_reports()
            logger.info(f"Loaded {len(self.reports)} reports")
        except Exception as e:
            logger.error(f"Error loading reports: {str(e)}")
            self.reports = {}
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process query using RAG"""
        if not self.reports:
            return "I'm sorry, but I couldn't load the project reports. Please ensure the Reports folder exists and contains PDF files."
        
        # Find relevant chunks
        relevant_chunks = self.pdf_processor.search_relevant_chunks(query, self.reports, top_k=5)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the project reports for your query. Would you like me to try answering from general knowledge instead?"
        
        # Build context from relevant chunks
        context_text = "\n\n---\n\n".join([
            f"From {chunk['report']}:\n{chunk['chunk']}"
            for chunk in relevant_chunks
        ])
        
        # Create prompt with context
        prompt = f"""Based on the following context from project reports, answer the user's question.

Context from Reports:
{context_text}

User Question: {query}

Provide a comprehensive answer based on the context. If the context doesn't fully answer the question, say so."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages)
        return response
