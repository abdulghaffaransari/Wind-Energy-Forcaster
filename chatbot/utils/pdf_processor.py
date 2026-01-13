"""
PDF Processing Utilities for RAG Agent
Extracts and processes text from PDF reports
"""
import PyPDF2
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files and extract text for RAG"""
    
    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def load_all_reports(self) -> Dict[str, str]:
        """Load all PDF reports from the Reports directory"""
        reports = {}
        
        if not self.reports_dir.exists():
            logger.warning(f"Reports directory not found: {self.reports_dir}")
            return reports
        
        pdf_files = list(self.reports_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            report_name = pdf_file.stem
            logger.info(f"Loading report: {report_name}")
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                reports[report_name] = text
        
        return reports
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def search_relevant_chunks(self, query: str, reports: Dict[str, str], top_k: int = 5) -> List[Dict]:
        """Find most relevant chunks for a query"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_chunks = []
        
        for report_name, text in reports.items():
            chunks = self.chunk_text(text)
            
            for chunk in chunks:
                chunk_lower = chunk.lower()
                chunk_words = set(chunk_lower.split())
                
                # Simple scoring: count matching words
                score = len(query_words.intersection(chunk_words))
                
                if score > 0:
                    scored_chunks.append({
                        "report": report_name,
                        "chunk": chunk,
                        "score": score
                    })
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
