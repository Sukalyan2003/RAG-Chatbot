"""
Final RAG System - Main Module

This module provides a comprehensive RAG (Retrieval-Augmented Generation) system
that combines the best features from multiple AI projects into a single, powerful,
and efficient chatbot system.

Features:
- Multi-format document processing
- Advanced embedding and retrieval
- Role-based access control
- Conversation memory
- Source attribution
- Streaming responses
- Comprehensive logging
"""

import json
import os
import logging
import asyncio
import re
from typing import Callable, Dict, List, Optional, Any, Union

# (stage, current, total) progress callback signature shared across the engine.
ProgressCallback = Callable[[str, Optional[int], Optional[int]], None]


def _safe_report(callback: Optional[ProgressCallback], stage: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
    """Invoke a progress callback while ignoring callback-side exceptions."""
    if callback is None:
        return
    try:
        callback(stage, current, total)
    except Exception:
        logger.debug("Progress callback raised; ignoring", exc_info=True)
from datetime import datetime, timedelta
import pickle
from pathlib import Path

# Import custom modules. The fallback keeps direct script execution working.
try:
    from .document_processor import DocumentProcessor
    from .embedding_manager import EmbeddingManager
    from .llm_interface import LLMInterface
    from .query_analyzer import QueryAnalyzer
    from .conversation_manager import ConversationManager
    from .utils import setup_logging, validate_input, sanitize_output
except ImportError:
    if __package__:
        raise
    from document_processor import DocumentProcessor
    from embedding_manager import EmbeddingManager
    from llm_interface import LLMInterface
    from query_analyzer import QueryAnalyzer
    from conversation_manager import ConversationManager
    from utils import setup_logging, validate_input, sanitize_output

logger = logging.getLogger(__name__)


class FinalRAGChatbot:
    """
    A comprehensive RAG chatbot system that provides intelligent document-based
    question answering with advanced features like role-based access control,
    conversation memory, and source attribution.
    """

    def __init__(self, role: str = "User", config_path: str = "config/config.json", custom_config: Optional[Dict] = None):
        """
        Initialize the Final RAG Chatbot system.
        
        Args:
            role: User role (Admin, User, Guest)
            config_path: Path to configuration file
            custom_config: Optional custom configuration dictionary
        """
        self.role = role
        self.config = self._load_config(config_path, custom_config)
        self.initialized = False
        self._start_time = datetime.now()
        self.last_error = ""
        self.last_load_summary = ""
        
        # Setup logging
        setup_logging(self.config["system"]["log_level"], self.config["paths"]["logs_dir"])
        logger.info(f"Initializing Final RAG Chatbot for role: {role}")
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.stats = {
            "queries_processed": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "errors": 0,
            "successful_responses": 0
        }
        
        logger.info("Final RAG Chatbot initialized successfully")

    def _load_config(self, config_path: str, custom_config: Optional[Dict]) -> Dict:
        """Load configuration from file and merge with custom config."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if custom_config:
                config.update(custom_config)
            
            # Ensure required directories exist
            for path_key, path_value in config["paths"].items():
                Path(path_value).mkdir(parents=True, exist_ok=True)
            
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize document processor
            self.document_processor = DocumentProcessor(self.config)
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(self.config)
            
            # Initialize LLM interface
            self.llm_interface = LLMInterface(self.config)
            
            # Initialize query analyzer
            self.query_analyzer = QueryAnalyzer(self.config, self.llm_interface)
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(self.config)
            
            # Load or create embeddings cache
            self._load_embeddings_cache()
            
            self.initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def _embeddings_cache_path(self) -> str:
        """Return the embeddings cache file path."""
        return os.path.join(self.config["paths"]["embeddings_dir"], "embeddings_cache.pkl")

    def _load_embeddings_cache(self):
        """Load existing embeddings cache or create new one."""
        cache_path = self._embeddings_cache_path()

        try:
            if os.path.exists(cache_path) and self.config["system"]["cache_embeddings"]:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                loaded = self.embedding_manager.load_cached_embeddings(cached_data)
                if loaded:
                    logger.info("Loaded embeddings from cache")
                else:
                    logger.warning(
                        "Embeddings cache rejected (model/dimension mismatch); "
                        "next ingest will rebuild against the configured embedding model."
                    )
            else:
                logger.info("No embeddings cache found, will create new one")
        except Exception as e:
            logger.warning(f"Error loading embeddings cache: {e}")

    def _save_embeddings_cache(self):
        """Save embeddings to cache."""
        if not self.config["system"]["cache_embeddings"]:
            return
            
        cache_path = self._embeddings_cache_path()
        
        try:
            cached_data = self.embedding_manager.get_cached_embeddings()
            if not cached_data.get("documents"):
                return
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)
            logger.info("Saved embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {e}")

    def _normalize_source(self, source: Union[str, List[str]]) -> Union[str, List[str]]:
        """Normalize user-entered file paths before document loading."""
        if isinstance(source, list):
            return [self._normalize_source(item) for item in source]

        normalized = source.strip().strip('"').strip("'")
        normalized = normalized.replace("\\ ", " ")
        normalized = os.path.expanduser(normalized)

        # Convert Windows paths pasted into WSL, e.g. C:\Users\Name\file.pdf.
        drive_match = re.match(r"^([A-Za-z]):[\\/](.*)", normalized)
        if drive_match and os.name != "nt":
            drive = drive_match.group(1).lower()
            rest = drive_match.group(2).replace("\\", "/")
            normalized = f"/mnt/{drive}/{rest}"

        return normalized

    def load_documents(
        self,
        source: Union[str, List[str]],
        document_type: str = "auto",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> bool:
        """
        Load documents from various sources.

        Args:
            source: File path, directory path, or list of paths
            document_type: Type of documents (auto, pdf, txt, json, csv, web)
            progress_callback: Optional ``(stage, current, total)`` callback. Stages
                are ``"reading"``, ``"chunking"``, ``"dedup"``, ``"embedding"``,
                ``"storing"``, and ``"saving_cache"``.

        Returns:
            bool: Success status
        """
        if not self.initialized:
            logger.error("System not initialized")
            self.last_error = "System is not initialized"
            return False

        try:
            self.last_error = ""
            self.last_load_summary = ""
            source = self._normalize_source(source)
            logger.info(f"Loading documents from: {source}")

            _safe_report(progress_callback, "reading", None, None)

            # Process documents
            documents = self.document_processor.process_documents(source, document_type)

            if not documents:
                self.last_error = (
                    f"No document chunks were processed from {source!r}. "
                    "Check that the path exists and the file type is supported."
                )
                logger.warning(self.last_error)
                return False

            _safe_report(progress_callback, "chunking", len(documents), len(documents))

            # Generate embeddings
            processed_sources = self._unique_document_sources(documents)
            before_sources = self._unique_document_sources(self.embedding_manager.documents)
            success = self.embedding_manager.add_documents(documents, progress_callback=progress_callback)
            
            if success:
                _safe_report(progress_callback, "saving_cache", None, None)
                self._save_embeddings_cache()
                added = getattr(self.embedding_manager, "last_added_count", len(documents))
                skipped = getattr(self.embedding_manager, "last_skipped_count", 0)
                added_sources = processed_sources - before_sources
                existing_sources = processed_sources & before_sources
                self.last_load_summary = (
                    f"Processed {self._plural(len(processed_sources), 'document')} "
                    f"from {source}; added {self._plural(len(added_sources), 'new document')}"
                )
                if existing_sources:
                    self.last_load_summary += f"; skipped {self._plural(len(existing_sources), 'already-loaded document')}"
                self.last_load_summary += f" ({added} new chunks, {skipped} duplicate chunks)"
                logger.info(self.last_load_summary)
                return True
            else:
                self.last_error = "Failed to generate embeddings for processed documents"
                logger.error(self.last_error)
                return False
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error loading documents: {e}")
            return False

    def chat(
        self,
        query: str,
        stream: bool = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Union[str, Dict]:
        """
        Process a chat query and return response.

        Args:
            query: User query
            stream: Whether to stream response (None = use config default)
            progress_callback: Optional ``(stage, current, total)`` callback. Stages
                fire in order: ``"validating"``, ``"analyzing"``, ``"retrieving"``,
                ``"context"``, ``"generating"``, ``"done"``. ``current``/``total``
                are ``None`` for indeterminate stages and chunk counts for
                ``"retrieving"`` (``retrieved`` count, ``max_results``).

        Returns:
            Response string or dictionary with streaming info
        """
        start_time = datetime.now()

        try:
            self.last_error = ""
            if hasattr(self.llm_interface, "last_error"):
                self.llm_interface.last_error = ""

            _safe_report(progress_callback, "validating", None, None)

            # Validate input
            if not validate_input(query, self.config):
                return "Sorry, your query contains invalid content. Please try again."

            # Check role permissions
            if not self._check_permissions(query):
                return "Sorry, you don't have permission to access this information."

            # Update statistics
            self.stats["queries_processed"] += 1

            logger.info(f"Processing query from {self.role}: {query[:100]}...")

            _safe_report(progress_callback, "analyzing", None, None)

            # Analyze query
            analyzed_query = self.query_analyzer.analyze_query(query, self.role)

            _safe_report(progress_callback, "retrieving", None, self.config["retrieval"]["max_results"])

            # Retrieve relevant documents
            relevant_docs = self.embedding_manager.retrieve_documents(
                analyzed_query,
                max_results=self.config["retrieval"]["max_results"],
                threshold=self.config["retrieval"]["similarity_threshold"]
            )

            _safe_report(progress_callback, "retrieving", len(relevant_docs), self.config["retrieval"]["max_results"])

            if not relevant_docs:
                _safe_report(progress_callback, "done", None, None)
                return "I couldn't find any relevant information to answer your question. Please try rephrasing or asking about something else."

            _safe_report(progress_callback, "context", None, None)

            # Get conversation context
            conversation_context = self.conversation_manager.get_context(self.role)

            # Generate response
            if stream is None:
                stream = self.config["system"]["enable_streaming"]

            _safe_report(progress_callback, "generating", None, None)

            if stream:
                response = self._generate_streaming_response(query, relevant_docs, conversation_context)
            else:
                response = self._generate_response(query, relevant_docs, conversation_context)

            _safe_report(progress_callback, "done", None, None)
            
            # Update conversation history
            self.conversation_manager.add_interaction(self.role, query, response, relevant_docs)
            
            # Update statistics
            response_time = (datetime.now() - start_time).total_seconds()
            llm_error = getattr(self.llm_interface, "last_error", "")
            if llm_error:
                self.last_error = llm_error
                self.stats["errors"] += 1
                self._update_stats(response_time, False)
            else:
                self._update_stats(response_time, True)
            
            logger.info(f"Query processed successfully in {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error processing query: {e}")
            self.stats["errors"] += 1
            return "I encountered an error while processing your request. Please try again."

    def _check_permissions(self, query: str) -> bool:
        """Check if user has permission to access requested information."""
        user_permissions = self.config["roles"].get(self.role, {}).get("permissions", [])
        
        # Simple keyword-based permission checking
        # In production, this should be more sophisticated
        if "financial" in query.lower() and "financial_data" not in user_permissions:
            return False
        
        return True

    def _generate_response(self, query: str, relevant_docs: List[Dict], context: List[Dict]) -> str:
        """Generate a standard response."""
        try:
            # Prepare context for LLM
            document_content = "\n\n".join([doc["content"] for doc in relevant_docs])
            conversation_history = "\n".join([
                f"Q: {item['query']}\nA: {self._strip_source_attribution(item['response'])}"
                for item in context[-3:]
            ])
            
            # Create system prompt based on role
            role_config = self.config["roles"].get(self.role, self.config["roles"]["User"])
            
            system_prompt = self._create_system_prompt(role_config)
            
            # Generate response
            response = self.llm_interface.generate_response(
                query=query,
                context=document_content,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )

            if getattr(self.llm_interface, "last_error", ""):
                return sanitize_output(response)
            
            response = self._clean_response_text(response)

            # Add source attribution if enabled
            if self.config["system"]["enable_source_attribution"]:
                response = self._strip_source_attribution(response)
                sources = self._format_sources(relevant_docs)
                if sources:
                    response += f"\n\n---\n\n**Sources**\n{sources}"
            
            return sanitize_output(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _generate_streaming_response(self, query: str, relevant_docs: List[Dict], context: List[Dict]) -> Dict:
        """Generate a streaming response."""
        # For now, return regular response with streaming metadata
        # In production, this would implement actual streaming
        response = self._generate_response(query, relevant_docs, context)
        
        return {
            "type": "streaming",
            "content": response,
            "sources": relevant_docs if self.config["system"]["enable_source_attribution"] else [],
            "metadata": {
                "role": self.role,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(relevant_docs)
            }
        }

    def _create_system_prompt(self, role_config: Dict) -> str:
        """Create system prompt based on role configuration."""
        base_prompt = f"""
        You are a helpful AI assistant with access to a knowledge base. 
        Your role permissions: {', '.join(role_config.get('permissions', []))}
        Response style: {role_config.get('response_length', 'concise')}
        
        Guidelines:
        - Always be accurate and helpful
        - Use the provided context to answer questions
        - If you don't know something, say so
        - Keep responses professional and respectful
        - Cite sources when possible
        - Use readable Markdown: short paragraphs, bullet lists, and tables when helpful
        - Do not append your own source list; the system adds sources separately
        """
        
        if role_config.get('response_length') == 'brief':
            base_prompt += "\n- Keep responses under 50 words unless explicitly asked for more detail"
        elif role_config.get('response_length') == 'detailed':
            base_prompt += "\n- Provide comprehensive and detailed responses with examples when appropriate"
        
        return base_prompt.strip()

    def _format_sources(self, relevant_docs: List[Dict]) -> str:
        """Format source attribution."""
        sources = []
        seen = set()
        for doc in relevant_docs:
            source = doc.get('metadata', {}).get('source', 'Unknown')
            if source in seen:
                continue
            seen.add(source)
            sources.append(f"- {self._display_source(source)}")
            if len(sources) >= 3:
                break
        
        return "\n".join(sources)

    def _clean_response_text(self, response: Any) -> str:
        """Clean provider-specific noise while keeping markdown formatting."""
        text = response.get("content", "") if isinstance(response, dict) else str(response)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    def _strip_source_attribution(self, response: Any) -> str:
        """Remove previously appended source blocks before reusing model output."""
        text = self._clean_response_text(response)
        text = re.split(r"\s+(?:---\s+)?(?:\*\*)?Sources(?:\*\*)?:?", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        text = re.sub(
            r"\n\s*\*{0,2}Source(?:s)?\s*:\s*.*?\*{0,2}\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    def _display_source(self, source: str) -> str:
        """Return a readable source label for responses and document lists."""
        if source.startswith(("http://", "https://")):
            return source
        return Path(source).name or source

    def _unique_document_sources(self, documents: List[Dict]) -> set:
        """Return unique source identifiers for document chunks."""
        return {
            doc.get("metadata", {}).get("source", "Unknown")
            for doc in documents
        }

    def _plural(self, count: int, label: str) -> str:
        """Format simple document/count labels."""
        suffix = "" if count == 1 else "s"
        return f"{count} {label}{suffix}"

    def get_document_summaries(self) -> List[Dict[str, Any]]:
        """Return one row per source document instead of one row per chunk."""
        summaries: Dict[str, Dict[str, Any]] = {}

        for doc in self.embedding_manager.documents:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            summary = summaries.setdefault(
                source,
                {
                    "document": self._display_source(source),
                    "type": metadata.get("type", "unknown"),
                    "chunks": 0,
                    "source": source,
                },
            )
            summary["chunks"] += 1

        return list(summaries.values())

    def _update_stats(self, response_time: float, success: bool):
        """Update performance statistics."""
        if success:
            self.stats["successful_responses"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_responses"]
            )

    def get_stats(self) -> Dict:
        """Get system performance statistics."""
        return {
            **self.stats,
            "documents_loaded": len(self.get_document_summaries()),
            "chunks_loaded": self.embedding_manager.get_document_count(),
            "conversation_history_length": self.conversation_manager.get_history_length(self.role),
            "system_uptime": str(datetime.now() - getattr(self, '_start_time', datetime.now())),
            "last_error": self.last_error
        }

    def clear_conversation(self):
        """Clear conversation history for current role."""
        self.conversation_manager.clear_history(self.role)
        logger.info(f"Cleared conversation history for {self.role}")

    def export_conversation(self, format: str = "json") -> Union[str, Dict]:
        """Export conversation history."""
        return self.conversation_manager.export_history(self.role, format)

    def reset_system(self):
        """Reset the entire system."""
        logger.info("Resetting system...")
        self.conversation_manager.clear_all_history()
        self.clear_documents(delete_cache=True)
        self.stats = {key: 0 for key in self.stats}
        logger.info("System reset complete")

    def clear_documents(self, delete_cache: bool = True) -> Dict[str, Any]:
        """Clear loaded documents, embeddings, and optionally the persisted cache file."""
        self.embedding_manager.clear_cache()
        cache_path = self._embeddings_cache_path()
        cache_deleted = False

        if delete_cache and os.path.exists(cache_path):
            os.remove(cache_path)
            cache_deleted = True

        self.last_error = ""
        logger.info("Cleared documents and embeddings")
        return {
            "documents_loaded": len(self.get_document_summaries()),
            "chunks_loaded": self.embedding_manager.get_document_count(),
            "cache_deleted": cache_deleted,
            "cache_path": cache_path,
        }

    async def chat_async(self, query: str) -> str:
        """Asynchronous version of chat method."""
        if not self.config["performance"]["enable_async"]:
            return self.chat(query)
        
        # Run in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, query)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        try:
            self._save_embeddings_cache()
            logger.info("Final RAG Chatbot shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final RAG Chatbot System")
    parser.add_argument("--role", default="User", choices=["Admin", "User", "Guest"])
    parser.add_argument("--config", default="config/config.json", help="Configuration file path")
    parser.add_argument("--documents", help="Documents to load (file or directory)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    try:
        with FinalRAGChatbot(role=args.role, config_path=args.config) as chatbot:
            
            # Load documents if specified
            if args.documents:
                print(f"Loading documents from: {args.documents}")
                if chatbot.load_documents(args.documents):
                    print("Documents loaded successfully!")
                else:
                    print("Failed to load documents")
                    return
            
            # Interactive mode
            if args.interactive:
                print(f"\n🤖 Final RAG Chatbot ({args.role} mode)")
                print("Type 'quit' to exit, 'stats' for statistics, 'clear' to clear conversation")
                print("-" * 50)
                
                while True:
                    try:
                        query = input(f"\n{args.role}: ").strip()
                        
                        if query.lower() == 'quit':
                            break
                        elif query.lower() == 'stats':
                            stats = chatbot.get_stats()
                            for key, value in stats.items():
                                print(f"{key}: {value}")
                            continue
                        elif query.lower() == 'clear':
                            chatbot.clear_conversation()
                            print("Conversation cleared!")
                            continue
                        
                        if query:
                            response = chatbot.chat(query)
                            print(f"\n🤖 Chatbot: {response}")
                        
                    except KeyboardInterrupt:
                        print("\n\nGoodbye!")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
            
            else:
                print("Final RAG Chatbot initialized successfully!")
                print("Use --interactive flag for interactive mode")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
