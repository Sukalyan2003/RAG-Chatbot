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
import hashlib
import logging
import pickle
from typing import Callable, List, Dict, Any, Optional, Tuple

try:
    from .bm25_index import BM25Index
    from .utils import reciprocal_rank_fusion
except ImportError:  # pragma: no cover - allows running as a top-level module
    from bm25_index import BM25Index
    from utils import reciprocal_rank_fusion

CACHE_SCHEMA_VERSION = 1

# Progress callback signature: (stage, current, total) where current/total may be None.
ProgressCallback = Callable[[str, Optional[int], Optional[int]], None]


def _safe_report(callback: Optional[ProgressCallback], stage: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
    """Invoke a progress callback while swallowing UI-side exceptions."""
    if callback is None:
        return
    try:
        callback(stage, current, total)
    except Exception:
        logger.debug("Progress callback raised; ignoring", exc_info=True)

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

        # Sparse lexical index kept in sync with self.documents for hybrid retrieval
        self.bm25 = BM25Index()

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

    def add_documents(
        self,
        documents: List[Dict],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> bool:
        """
        Add documents and generate embeddings.

        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            progress_callback: Optional ``(stage, current, total)`` callback invoked
                during dedup and per-batch embedding. ``current`` and ``total`` are
                chunk counts; ``stage`` is one of ``"dedup"``, ``"embedding"``,
                ``"storing"``.

        Returns:
            bool: Success status
        """
        try:
            self.last_added_count = 0
            self.last_skipped_count = 0

            if not documents:
                logger.warning("No documents provided")
                return False

            _safe_report(progress_callback, "dedup", 0, len(documents))

            # Stamp content_hash onto incoming docs so dedup is hash-based
            for doc in documents:
                metadata = doc.setdefault("metadata", {})
                if "content_hash" not in metadata:
                    metadata["content_hash"] = self._content_hash(doc.get("content", ""))

            existing_keys = {self._document_key(doc) for doc in self.documents}
            new_documents = [
                doc for doc in documents
                if self._document_key(doc) not in existing_keys
            ]
            self.last_skipped_count = len(documents) - len(new_documents)

            if not new_documents:
                logger.info("All provided documents are already loaded; skipping duplicates")
                _safe_report(progress_callback, "embedding", 0, 0)
                return True

            documents = new_documents

            logger.info(f"Adding {len(documents)} documents")

            # Extract text content
            texts = [doc["content"] for doc in documents]

            # Generate embeddings in batches
            logger.info("Generating embeddings...")
            _safe_report(progress_callback, "embedding", 0, len(texts))
            new_embeddings = self._generate_embeddings_batch(texts, progress_callback=progress_callback)
            
            if new_embeddings is None:
                logger.error("Failed to generate embeddings")
                return False

            _safe_report(progress_callback, "storing", len(documents), len(documents))

            # Add to storage
            self.documents.extend(documents)
            self.bm25.add([(self._chunk_id(doc), doc["content"]) for doc in documents])

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
        """Return a stable key for duplicate document chunk detection.

        The key uses a SHA-256 content hash instead of the full content string so
        repeated ingests of the same chunk are an O(N) hash lookup rather than an
        O(N*L) string compare.
        """
        metadata = doc.get("metadata", {})
        chunk_marker = (
            metadata.get("chunk_id"),
            metadata.get("item_id"),
            metadata.get("row_id"),
        )
        content_hash = metadata.get("content_hash") or self._content_hash(doc.get("content", ""))
        return (
            metadata.get("source", ""),
            metadata.get("file_name", ""),
            metadata.get("type", ""),
            chunk_marker,
            content_hash,
        )

    def _rebuild_bm25(self) -> None:
        """Rebuild the sparse index from scratch to match ``self.documents``."""
        self.bm25.clear()
        self.bm25.add([(self._chunk_id(doc), doc.get("content", "")) for doc in self.documents])

    def _chunk_id(self, doc: Dict) -> str:
        """Stable string id for a chunk, shared by the dense and BM25 legs.

        Derived from ``_document_key`` so it matches whether the chunk is freshly
        added or rehydrated from cache, and stays unique across sources that
        happen to share identical content.
        """
        return repr(self._document_key(doc))

    @staticmethod
    def _content_hash(content: str) -> str:
        """Truncated SHA-256 hex digest used as a chunk identifier."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _generate_embeddings_batch(
        self,
        texts: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            if self.provider == "ollama":
                return self._generate_ollama_embeddings(texts, progress_callback=progress_callback)

            # Split into batches to avoid memory issues
            all_embeddings = []
            total = len(texts)

            for i in range(0, total, self.batch_size):
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
                _safe_report(progress_callback, "embedding", min(i + len(batch), total), total)

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def _generate_ollama_embeddings(
        self,
        texts: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[np.ndarray]:
        """Generate embeddings using Ollama's native embedding API."""
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = [
                text[:self.max_length] if len(text) > self.max_length else text
                for text in texts[i:i + self.batch_size]
            ]

            embeddings = self._call_ollama_embed(batch)
            if embeddings is None:
                return None

            all_embeddings.extend(embeddings)
            _safe_report(progress_callback, "embedding", min(i + len(batch), total), total)

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

        With ``retrieval.hybrid_enabled`` (and ``rank_bm25`` installed), a dense
        cosine ranking and a BM25 lexical ranking are fused with reciprocal rank
        fusion to form the candidate set; otherwise a dense-only, threshold-gated
        ranking is used. When ``retrieval.rerank_results`` is enabled the
        oversampled candidate set (``max_results * rerank_oversample_factor``) is
        passed through ``rerank_results`` and trimmed to ``max_results``.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            threshold: Minimum similarity threshold (dense-only path)

        Returns:
            List of relevant documents with similarity scores
        """
        try:
            if self.embeddings is None or len(self.documents) == 0:
                logger.warning("No documents available for retrieval")
                return []

            retrieval_config = self.config.get("retrieval", {})
            rerank_enabled = retrieval_config.get("rerank_results", False)
            oversample_factor = retrieval_config.get("rerank_oversample_factor", 4)
            hybrid_enabled = retrieval_config.get("hybrid_enabled", False)

            candidate_limit = max_results * oversample_factor if rerank_enabled else max_results * 2

            # Generate query embedding and dense cosine similarities (needed by
            # both paths; also used as the displayed similarity_score in hybrid).
            query_embedding = self._embed_query(query)
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            if hybrid_enabled and self.bm25.available:
                candidates = self._hybrid_candidates(
                    query, similarities, retrieval_config, candidate_limit
                )
                mode = "hybrid"
            else:
                candidates = self._dense_candidates(
                    similarities, threshold, candidate_limit
                )
                mode = "dense"

            if rerank_enabled and candidates:
                candidates = self.rerank_results(candidates, query)

            relevant_docs = candidates[:max_results]

            logger.info(
                "Retrieved %s relevant documents for query (mode=%s, candidates=%s, rerank=%s)",
                len(relevant_docs), mode, len(candidates), rerank_enabled,
            )
            return relevant_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _dense_candidates(
        self, similarities, threshold: float, candidate_limit: int
    ) -> List[Dict]:
        """Dense-only candidates: threshold-gated cosine top-K."""
        sorted_indices = np.argsort(similarities)[::-1]
        candidates = []
        for idx in sorted_indices[:candidate_limit]:
            similarity = similarities[idx]
            if similarity >= threshold:
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(similarity)
                candidates.append(doc)
        return candidates

    def _hybrid_candidates(
        self, query: str, similarities, retrieval_config: Dict, candidate_limit: int
    ) -> List[Dict]:
        """Fuse dense and BM25 rankings with reciprocal rank fusion.

        The dense leg and the BM25 leg each contribute a ranked list of chunk
        ids; RRF combines them and the top ``candidate_limit`` fused chunks are
        materialized as documents (carrying their dense cosine ``similarity_score``
        for display and the ``fusion_score`` used for ordering). Unlike the
        dense-only path this does not apply the cosine threshold, so chunks that
        match lexically but embed poorly still surface.
        """
        dense_top_k = retrieval_config.get("dense_top_k", 20)
        bm25_top_k = retrieval_config.get("bm25_top_k", 20)
        rrf_k = retrieval_config.get("rrf_k", 60)

        chunk_ids = [self._chunk_id(doc) for doc in self.documents]
        pos_for_id = {}
        for position, chunk_id in enumerate(chunk_ids):
            pos_for_id.setdefault(chunk_id, position)

        dense_order = np.argsort(similarities)[::-1][:dense_top_k]
        dense_ranking = [chunk_ids[idx] for idx in dense_order]
        bm25_ranking = [chunk_id for chunk_id, _ in self.bm25.query(query, bm25_top_k)]

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=rrf_k)

        candidates = []
        for chunk_id, fusion_score in fused[:candidate_limit]:
            position = pos_for_id.get(chunk_id)
            if position is None:
                continue
            doc = self.documents[position].copy()
            doc["similarity_score"] = float(similarities[position])
            doc["fusion_score"] = float(fusion_score)
            candidates.append(doc)
        return candidates

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
            old_chunk_id = self._chunk_id(self.documents[document_index])
            self.documents[document_index]["content"] = new_content
            self.documents[document_index].get("metadata", {}).pop("content_hash", None)

            # Regenerate embedding
            new_embedding = self._embed_query(new_content)
            self.embeddings[document_index] = new_embedding[0]

            # Keep the sparse index in sync with the new content
            self.bm25.remove([old_chunk_id])
            self.bm25.add([(self._chunk_id(self.documents[document_index]), new_content)])
            
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
            removed_chunk_id = self._chunk_id(self.documents[document_index])
            del self.documents[document_index]
            self.bm25.remove([removed_chunk_id])

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

    def load_cached_embeddings(self, cached_data: Dict) -> bool:
        """Load embeddings from cache.

        Refuses to load when the cache header reports a different embedding
        model or vector dimension from the currently configured model. On
        mismatch the in-memory state is left empty so the next ``add_documents``
        call rebuilds against the configured model. Returns ``True`` when the
        cache was loaded, ``False`` when it was rejected or empty.
        """
        try:
            cached_model = cached_data.get("model_name")
            if cached_model and cached_model != self.embedding_model_name:
                logger.warning(
                    "Embeddings cache was built with model %r but configuration uses %r; "
                    "ignoring cache and rebuilding.",
                    cached_model,
                    self.embedding_model_name,
                )
                self.documents = []
                self.embeddings = None
                self.bm25.clear()
                return False

            embeddings_data = cached_data.get("embeddings")
            cached_dim = cached_data.get("dim")
            if embeddings_data is not None and cached_dim is not None:
                first_row = embeddings_data[0] if embeddings_data else None
                actual_dim = len(first_row) if first_row is not None else None
                if actual_dim is not None and actual_dim != cached_dim:
                    logger.warning(
                        "Embeddings cache header reports dim=%s but payload has dim=%s; "
                        "ignoring cache and rebuilding.",
                        cached_dim,
                        actual_dim,
                    )
                    self.documents = []
                    self.embeddings = None
                    return False

            self.documents = cached_data.get("documents", [])

            if embeddings_data is not None:
                self.embeddings = np.array(embeddings_data)

            self._rebuild_bm25()

            logger.info(f"Loaded {len(self.documents)} documents from cache")
            return True

        except Exception as e:
            logger.error(f"Error loading cached embeddings: {e}")
            self.documents = []
            self.embeddings = None
            self.bm25.clear()
            return False

    def get_cached_embeddings(self) -> Dict:
        """Get embeddings data for caching, including a versioned header."""
        embeddings_list = self.embeddings.tolist() if self.embeddings is not None else None
        dim = None
        if embeddings_list:
            first_row = embeddings_list[0]
            dim = len(first_row) if first_row is not None else None
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_name": self.embedding_model_name,
            "dim": dim,
            "documents": self.documents,
            "embeddings": embeddings_list,
        }

    def clear_cache(self):
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        self.bm25.clear()
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
