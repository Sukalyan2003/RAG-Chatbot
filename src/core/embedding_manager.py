from __future__ import annotations

"""
Embedding Manager Module

This module handles all embedding-related operations including:
- Text embedding generation using sentence transformers
- Vector storage and retrieval
- Similarity search and ranking
- Embedding caching for performance
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    NUMERIC_AVAILABLE = True
except ImportError:
    np = None
    cosine_similarity = None
    NUMERIC_AVAILABLE = False
    logging.warning("numpy/scikit-learn not available")

# Sentence transformers for embeddings
SentenceTransformer = None
TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available")

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages document embeddings, vector storage, and similarity search.
    """

    def __init__(self, config: Dict):
        """
        Initialize the embedding manager.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.embedding_config = config["embedding"]
        self.provider = self.embedding_config.get("provider", "sentence_transformers")
        self.embedding_model_name = self.embedding_config["model"]
        self.batch_size = self.embedding_config["batch_size"]
        self.max_length = self.embedding_config["max_length"]
        self.device = self.embedding_config.get("device", "cpu")
        self.base_url = self.embedding_config.get(
            "base_url",
            config.get("llm", {}).get("base_url", "http://localhost:11434")
        ).rstrip("/")
        self.timeout = self.embedding_config.get("timeout", config.get("llm", {}).get("timeout", 30))
        
        # Storage for documents and embeddings
        self.documents = []
        self.embeddings = None
        self.embedding_model = None
        self.last_added_count = 0
        self.last_skipped_count = 0
        
        # Initialize embedding model
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the configured embedding provider."""
        global SentenceTransformer, TRANSFORMERS_AVAILABLE

        if not NUMERIC_AVAILABLE:
            logger.error("numpy and scikit-learn are not available")
            raise ImportError("numpy and scikit-learn packages required")

        if self.provider == "ollama":
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests package required for Ollama embeddings")
            logger.info(f"Using Ollama embedding model: {self.embedding_model_name}")
            return

        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer

                SentenceTransformer = _SentenceTransformer
                TRANSFORMERS_AVAILABLE = True
            except ImportError:
                TRANSFORMERS_AVAILABLE = False

        if self.provider != "sentence_transformers":
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        if not TRANSFORMERS_AVAILABLE:
            logger.error("Sentence transformers not available")
            raise ImportError("sentence-transformers package required")
        
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def add_documents(self, documents: List[Dict]) -> bool:
        """
        Add documents and generate embeddings.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            bool: Success status
        """
        try:
            self.last_added_count = 0
            self.last_skipped_count = 0

            if not documents:
                logger.warning("No documents provided")
                return False

            existing_keys = {self._document_key(doc) for doc in self.documents}
            new_documents = [
                doc for doc in documents
                if self._document_key(doc) not in existing_keys
            ]
            self.last_skipped_count = len(documents) - len(new_documents)

            if not new_documents:
                logger.info("All provided documents are already loaded; skipping duplicates")
                return True

            documents = new_documents
            
            logger.info(f"Adding {len(documents)} documents")
            
            # Extract text content
            texts = [doc["content"] for doc in documents]
            
            # Generate embeddings in batches
            logger.info("Generating embeddings...")
            new_embeddings = self._generate_embeddings_batch(texts)
            
            if new_embeddings is None:
                logger.error("Failed to generate embeddings")
                return False
            
            # Add to storage
            self.documents.extend(documents)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            logger.info(f"Successfully added {len(documents)} documents. Total: {len(self.documents)}")
            self.last_added_count = len(documents)
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def _document_key(self, doc: Dict) -> Tuple:
        """Return a stable key for duplicate document chunk detection."""
        metadata = doc.get("metadata", {})
        chunk_marker = (
            metadata.get("chunk_id"),
            metadata.get("item_id"),
            metadata.get("row_id"),
        )
        return (
            metadata.get("source", ""),
            metadata.get("file_name", ""),
            metadata.get("type", ""),
            chunk_marker,
            doc.get("content", ""),
        )

    def _generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            if self.provider == "ollama":
                return self._generate_ollama_embeddings(texts)

            # Split into batches to avoid memory issues
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Truncate texts if they're too long
                truncated_batch = [
                    text[:self.max_length] if len(text) > self.max_length else text
                    for text in batch
                ]
                
                batch_embeddings = self.embedding_model.encode(
                    truncated_batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def _generate_ollama_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using Ollama's native embedding API."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = [
                text[:self.max_length] if len(text) > self.max_length else text
                for text in texts[i:i + self.batch_size]
            ]

            embeddings = self._call_ollama_embed(batch)
            if embeddings is None:
                return None

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def _call_ollama_embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call Ollama embedding endpoints, supporting current and legacy shapes."""
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embedding_model_name, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if "embeddings" in data:
                return data["embeddings"]
            if "embedding" in data:
                return [data["embedding"]]

        except Exception as e:
            logger.debug(f"Ollama /api/embed failed, trying legacy endpoint: {e}")

        embeddings = []
        try:
            for text in texts:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embedding_model_name, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if embedding is None:
                    raise ValueError("Ollama response did not include an embedding")
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error calling Ollama embeddings API: {e}")
            return None

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate a single query embedding with the configured provider."""
        if self.provider == "ollama":
            embedding = self._generate_ollama_embeddings([query])
            if embedding is None:
                raise RuntimeError("Failed to generate Ollama query embedding")
            return embedding

        return self.embedding_model.encode([query], convert_to_numpy=True)

    def retrieve_documents(self, query: str, max_results: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant documents based on query similarity.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant documents with similarity scores
        """
        try:
            if self.embeddings is None or len(self.documents) == 0:
                logger.warning("No documents available for retrieval")
                return []
            
            # Generate query embedding
            query_embedding = self._embed_query(query)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get indices sorted by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Filter by threshold and limit results
            relevant_docs = []
            
            for idx in sorted_indices[:max_results * 2]:  # Get more than needed in case of filtering
                similarity = similarities[idx]
                
                if similarity >= threshold:
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(similarity)
                    relevant_docs.append(doc)
                    
                    if len(relevant_docs) >= max_results:
                        break
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def get_similar_documents(self, document_index: int, max_results: int = 5) -> List[Dict]:
        """
        Find documents similar to a specific document.
        
        Args:
            document_index: Index of the source document
            max_results: Maximum number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            if document_index >= len(self.documents) or self.embeddings is None:
                logger.warning("Invalid document index or no embeddings available")
                return []
            
            doc_embedding = self.embeddings[document_index:document_index+1]
            similarities = cosine_similarity(doc_embedding, self.embeddings)[0]
            
            # Exclude the document itself
            similarities[document_index] = -1
            
            sorted_indices = np.argsort(similarities)[::-1]
            
            similar_docs = []
            for idx in sorted_indices[:max_results]:
                if similarities[idx] > 0:  # Only positive similarities
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(similarities[idx])
                    similar_docs.append(doc)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def update_document(self, document_index: int, new_content: str) -> bool:
        """
        Update a document and regenerate its embedding.
        
        Args:
            document_index: Index of document to update
            new_content: New content for the document
            
        Returns:
            bool: Success status
        """
        try:
            if document_index >= len(self.documents):
                logger.error("Invalid document index")
                return False
            
            # Update document content
            self.documents[document_index]["content"] = new_content
            
            # Regenerate embedding
            new_embedding = self._embed_query(new_content)
            self.embeddings[document_index] = new_embedding[0]
            
            logger.info(f"Updated document {document_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    def remove_document(self, document_index: int) -> bool:
        """
        Remove a document and its embedding.
        
        Args:
            document_index: Index of document to remove
            
        Returns:
            bool: Success status
        """
        try:
            if document_index >= len(self.documents):
                logger.error("Invalid document index")
                return False
            
            # Remove document
            del self.documents[document_index]
            
            # Remove embedding
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, document_index, axis=0)
            
            logger.info(f"Removed document {document_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False

    def search_by_metadata(self, metadata_filter: Dict) -> List[int]:
        """
        Search documents by metadata criteria.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match
            
        Returns:
            List of document indices matching the criteria
        """
        matching_indices = []
        
        for i, doc in enumerate(self.documents):
            metadata = doc.get("metadata", {})
            
            # Check if all filter criteria match
            match = True
            for key, value in metadata_filter.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                matching_indices.append(i)
        
        return matching_indices

    def get_embedding_statistics(self) -> Dict:
        """Get statistics about the embeddings."""
        if self.embeddings is None:
            return {"status": "No embeddings available"}
        
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embeddings.shape[1],
            "memory_usage_mb": self.embeddings.nbytes / (1024 * 1024),
            "model_name": self.embedding_model_name
        }

    def load_cached_embeddings(self, cached_data: Dict):
        """Load embeddings from cache."""
        try:
            self.documents = cached_data.get("documents", [])
            embeddings_data = cached_data.get("embeddings")
            
            if embeddings_data is not None:
                self.embeddings = np.array(embeddings_data)
            
            logger.info(f"Loaded {len(self.documents)} documents from cache")
            
        except Exception as e:
            logger.error(f"Error loading cached embeddings: {e}")

    def get_cached_embeddings(self) -> Dict:
        """Get embeddings data for caching."""
        return {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "model_name": self.embedding_model_name
        }

    def clear_cache(self):
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        logger.info("Cleared embedding cache")

    def get_document_count(self) -> int:
        """Get the number of stored documents."""
        return len(self.documents)

    def get_document_by_index(self, index: int) -> Optional[Dict]:
        """Get a document by its index."""
        if 0 <= index < len(self.documents):
            return self.documents[index]
        return None

    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank search results using additional criteria.
        
        Args:
            results: List of search results
            query: Original search query
            
        Returns:
            Re-ranked results
        """
        if not self.config["retrieval"]["rerank_results"]:
            return results
        
        try:
            # Simple re-ranking based on content length and metadata
            def rank_score(doc):
                base_score = doc.get("similarity_score", 0)
                
                # Prefer longer, more detailed content
                content_length_bonus = min(len(doc["content"]) / 1000, 0.1)
                
                # Prefer certain document types
                doc_type = doc.get("metadata", {}).get("type", "")
                type_bonus = {"pdf": 0.05, "text": 0.03, "web": 0.02}.get(doc_type, 0)
                
                return base_score + content_length_bonus + type_bonus
            
            # Sort by combined score
            results.sort(key=rank_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error re-ranking results: {e}")
            return results

    def export_embeddings(self, file_path: str) -> bool:
        """
        Export embeddings and documents to a file.
        
        Args:
            file_path: Path to save the embeddings
            
        Returns:
            bool: Success status
        """
        try:
            export_data = self.get_cached_embeddings()
            
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f)
            
            logger.info(f"Exported embeddings to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting embeddings: {e}")
            return False

    def import_embeddings(self, file_path: str) -> bool:
        """
        Import embeddings and documents from a file.
        
        Args:
            file_path: Path to load the embeddings from
            
        Returns:
            bool: Success status
        """
        try:
            with open(file_path, 'rb') as f:
                import_data = pickle.load(f)
            
            self.load_cached_embeddings(import_data)
            logger.info(f"Imported embeddings from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing embeddings: {e}")
            return False

    def get_all_documents(self) -> List[Dict]:
        """
        Get all documents stored in the embedding manager.
        
        Returns:
            List of all documents
        """
        return self.documents.copy()

    def find_documents_by_content(self, search_text: str, case_sensitive: bool = False) -> List[int]:
        """
        Find documents containing specific text.
        
        Args:
            search_text: Text to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            List of document indices containing the text
        """
        matching_indices = []
        
        if not case_sensitive:
            search_text = search_text.lower()
        
        for i, doc in enumerate(self.documents):
            content = doc.get("content", "")
            if not case_sensitive:
                content = content.lower()
            
            if search_text in content:
                matching_indices.append(i)
        
        return matching_indices
