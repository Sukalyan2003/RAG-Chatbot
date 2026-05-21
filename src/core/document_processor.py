"""
Document Processor Module

This module handles the processing of various document types including:
- PDF files
- Text files
- JSON files
- CSV files
- Web content
- Markdown files

It provides intelligent text chunking and metadata extraction.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Union
from pathlib import Path
import re

# PDF processing
try:
    from pdfminer.high_level import extract_text
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pdfminer not available, PDF processing disabled")

# DOCX processing
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Web scraping
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    logging.warning("requests/bs4 not available, web scraping disabled")

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles processing of various document types into standardized format
    for embedding and retrieval.
    """

    def __init__(self, config: Dict):
        """
        Initialize the document processor.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.chunk_size = config["retrieval"]["chunk_size"]
        self.chunk_overlap = config["retrieval"]["chunk_overlap"]
        
        # Supported file extensions
        self.supported_extensions = {
            '.txt': self._process_text,
            '.md': self._process_text,
            '.json': self._process_json,
            '.csv': self._process_csv,
        }
        
        if PDF_AVAILABLE:
            self.supported_extensions['.pdf'] = self._process_pdf
        if DOCX_AVAILABLE:
            self.supported_extensions['.docx'] = self._process_docx

    def process_documents(self, source: Union[str, List[str]], document_type: str = "auto") -> List[Dict]:
        """
        Process documents from various sources.
        
        Args:
            source: File path, directory path, URL, or list of paths
            document_type: Type of documents (auto, pdf, txt, json, csv, web)
            
        Returns:
            List of processed document dictionaries
        """
        documents = []
        
        try:
            if isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # Web URL
                    documents.extend(self._process_web_url(source))
                elif os.path.isfile(source):
                    # Single file
                    doc = self._process_single_file(source, document_type)
                    if doc:
                        documents.extend(doc)
                elif os.path.isdir(source):
                    # Directory
                    documents.extend(self._process_directory(source, document_type))
                else:
                    logger.error(f"Source not found: {source}")
            
            elif isinstance(source, list):
                # List of sources
                for item in source:
                    documents.extend(self.process_documents(item, document_type))
            
            logger.info(f"Processed {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return []

    def _process_single_file(self, file_path: str, document_type: str = "auto") -> List[Dict]:
        """Process a single file."""
        try:
            if Path(file_path).name.startswith("."):
                return []

            file_ext = Path(file_path).suffix.lower()
            
            if document_type == "auto":
                if file_ext not in self.supported_extensions:
                    logger.warning(f"Unsupported file type for {file_path}: {file_ext or 'no extension'}")
                    return []
                processor = self.supported_extensions[file_ext]
            else:
                # Use specified processor
                processor_map = {
                    'pdf': self._process_pdf,
                    'txt': self._process_text,
                    'json': self._process_json,
                    'csv': self._process_csv,
                    'docx': self._process_docx,
                }
                processor = processor_map.get(document_type, self._process_text)
            
            return processor(file_path)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    def _process_directory(self, dir_path: str, document_type: str = "auto") -> List[Dict]:
        """Process all supported files in a directory."""
        documents = []
        
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    docs = self._process_single_file(file_path, document_type)
                    documents.extend(docs)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")
            return []

    def _process_text(self, file_path: str) -> List[Dict]:
        """Process text/markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = self._split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': file_path,
                        'type': 'text',
                        'chunk_id': i,
                        'file_name': os.path.basename(file_path)
                    }
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def _process_pdf(self, file_path: str) -> List[Dict]:
        """Process PDF files."""
        if not PDF_AVAILABLE:
            logger.error("PDF processing not available")
            return []
            
        try:
            content = extract_text(file_path)
            chunks = self._split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': file_path,
                        'type': 'pdf',
                        'chunk_id': i,
                        'file_name': os.path.basename(file_path)
                    }
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return []

    def _process_docx(self, file_path: str) -> List[Dict]:
        """Process DOCX files."""
        if not DOCX_AVAILABLE:
            logger.error("DOCX processing not available")
            return []

        try:
            document = DocxDocument(file_path)
            paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            content = "\n".join(paragraphs)
            chunks = self._split_text(content)

            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': file_path,
                        'type': 'docx',
                        'chunk_id': i,
                        'file_name': os.path.basename(file_path)
                    }
                })

            return documents

        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {e}")
            return []

    def _process_json(self, file_path: str) -> List[Dict]:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    content = self._extract_content_from_json_item(item)
                    if content:
                        documents.append({
                            'content': content,
                            'metadata': {
                                'source': file_path,
                                'type': 'json',
                                'item_id': i,
                                'file_name': os.path.basename(file_path)
                            }
                        })
            
            elif isinstance(data, dict):
                content = self._extract_content_from_json_item(data)
                if content:
                    chunks = self._split_text(content)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {
                                'source': file_path,
                                'type': 'json',
                                'chunk_id': i,
                                'file_name': os.path.basename(file_path)
                            }
                        })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            return []

    def _process_csv(self, file_path: str) -> List[Dict]:
        """Process CSV files."""
        try:
            documents = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                
                for i, row in enumerate(csv_reader):
                    # Convert row to text
                    content = self._format_csv_row(row)
                    
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': file_path,
                            'type': 'csv',
                            'row_id': i,
                            'file_name': os.path.basename(file_path)
                        }
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            return []

    def _process_web_url(self, url: str) -> List[Dict]:
        """Process web content from URL."""
        if not WEB_AVAILABLE:
            logger.error("Web scraping not available")
            return []
            
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": (
                        "FinalRAGChatbot/1.0 "
                        "(local document processing example; educational use)"
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            chunks = self._split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': url,
                        'type': 'web',
                        'chunk_id': i,
                        'title': soup.title.string if soup.title else 'Unknown'
                    }
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing web URL {url}: {e}")
            return []

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        Uses intelligent splitting at sentence boundaries when possible.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_words = current_chunk.split()[-self.chunk_overlap:]
                        current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    chunks.extend(self._split_long_sentence(sentence))
                    current_chunk = ""
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that's longer than chunk size."""
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    # Single word is too long
                    chunks.append(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _extract_content_from_json_item(self, item: Any) -> str:
        """Extract meaningful content from a JSON item."""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Look for common content fields
            content_fields = ['content', 'text', 'body', 'message', 'description', 'summary']
            
            for field in content_fields:
                if field in item and isinstance(item[field], str):
                    return item[field]
            
            # If no standard content field, join all string values
            text_parts = []
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:  # Skip short values like IDs
                    text_parts.append(f"{key}: {value}")
            
            return '\n'.join(text_parts)
        
        else:
            return str(item)

    def _format_csv_row(self, row: Dict) -> str:
        """Format a CSV row as readable text."""
        parts = []
        for key, value in row.items():
            if value and str(value).strip():
                parts.append(f"{key}: {value}")
        
        return '\n'.join(parts)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        formats = list(self.supported_extensions.keys())
        if WEB_AVAILABLE:
            formats.append('web_url')
        return formats
