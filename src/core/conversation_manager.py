"""
Conversation Manager Module

This module handles conversation history, context management, and user session tracking.
It provides features for:
- Maintaining conversation history per user role
- Context-aware conversation flow
- Session management and timeout handling
- Conversation export and analysis
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context for different user roles.
    """

    def __init__(self, config: Dict):
        """
        Initialize the conversation manager.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.max_history = config["system"]["max_conversation_history"]
        self.conversation_timeout = config["system"]["conversation_timeout"]
        
        # Storage for conversation histories by role
        self.conversations = defaultdict(list)
        self.session_metadata = defaultdict(dict)
        
        logger.info("Conversation manager initialized")

    def add_interaction(self, role: str, query: str, response: str, sources: List[Dict] = None):
        """
        Add a new interaction to the conversation history.
        
        Args:
            role: User role
            query: User query
            response: System response
            sources: Source documents used (optional)
        """
        try:
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "sources": sources or [],
                "metadata": {
                    "query_length": len(query),
                    "response_length": len(response),
                    "source_count": len(sources) if sources else 0
                }
            }
            
            # Add to conversation history
            self.conversations[role].append(interaction)
            
            # Maintain maximum history length
            if len(self.conversations[role]) > self.max_history:
                self.conversations[role] = self.conversations[role][-self.max_history:]
            
            # Update session metadata
            self._update_session_metadata(role)
            
            logger.debug(f"Added interaction for role {role}")
            
        except Exception as e:
            logger.error(f"Error adding interaction: {e}")

    def get_context(self, role: str, context_length: int = 3) -> List[Dict]:
        """
        Get recent conversation context for a role.
        
        Args:
            role: User role
            context_length: Number of recent interactions to include
            
        Returns:
            List of recent interactions
        """
        try:
            # Check for session timeout
            if self._is_session_expired(role):
                logger.info(f"Session expired for role {role}, clearing history")
                self.clear_history(role)
                return []
            
            # Return recent interactions
            recent_interactions = self.conversations[role][-context_length:]
            
            logger.debug(f"Retrieved {len(recent_interactions)} context items for role {role}")
            return recent_interactions
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []

    def get_full_history(self, role: str) -> List[Dict]:
        """
        Get the complete conversation history for a role.
        
        Args:
            role: User role
            
        Returns:
            Complete conversation history
        """
        try:
            if self._is_session_expired(role):
                logger.info(f"Session expired for role {role}")
                self.clear_history(role)
                return []
            
            return self.conversations[role].copy()
            
        except Exception as e:
            logger.error(f"Error getting full history: {e}")
            return []

    def clear_history(self, role: str):
        """
        Clear conversation history for a specific role.
        
        Args:
            role: User role to clear history for
        """
        try:
            self.conversations[role] = []
            self.session_metadata[role] = {
                "session_start": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "interaction_count": 0
            }
            
            logger.info(f"Cleared conversation history for role {role}")
            
        except Exception as e:
            logger.error(f"Error clearing history: {e}")

    def clear_all_history(self):
        """Clear all conversation histories."""
        try:
            self.conversations.clear()
            self.session_metadata.clear()
            logger.info("Cleared all conversation histories")
            
        except Exception as e:
            logger.error(f"Error clearing all history: {e}")

    def get_history_length(self, role: str) -> int:
        """
        Get the length of conversation history for a role.
        
        Args:
            role: User role
            
        Returns:
            Number of interactions in history
        """
        return len(self.conversations.get(role, []))

    def get_conversation_summary(self, role: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation for a role.
        
        Args:
            role: User role
            
        Returns:
            Conversation summary statistics
        """
        try:
            history = self.conversations.get(role, [])
            
            if not history:
                return {"status": "No conversation history"}
            
            # Calculate statistics
            total_interactions = len(history)
            total_query_chars = sum(len(item["query"]) for item in history)
            total_response_chars = sum(len(item["response"]) for item in history)
            total_sources = sum(len(item.get("sources", [])) for item in history)
            
            # Time span
            first_interaction = datetime.fromisoformat(history[0]["timestamp"])
            last_interaction = datetime.fromisoformat(history[-1]["timestamp"])
            time_span = last_interaction - first_interaction
            
            summary = {
                "role": role,
                "total_interactions": total_interactions,
                "average_query_length": total_query_chars / total_interactions if total_interactions > 0 else 0,
                "average_response_length": total_response_chars / total_interactions if total_interactions > 0 else 0,
                "total_sources_used": total_sources,
                "conversation_span_minutes": time_span.total_seconds() / 60,
                "session_metadata": self.session_metadata.get(role, {})
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {"error": str(e)}

    def export_history(self, role: str, format: str = "json") -> str:
        """
        Export conversation history in specified format.
        
        Args:
            role: User role
            format: Export format (json, txt, csv)
            
        Returns:
            Formatted conversation data
        """
        try:
            history = self.conversations.get(role, [])
            
            if format.lower() == "json":
                return json.dumps({
                    "role": role,
                    "export_timestamp": datetime.now().isoformat(),
                    "conversation_history": history,
                    "summary": self.get_conversation_summary(role)
                }, indent=2)
            
            elif format.lower() == "txt":
                lines = [f"Conversation History for {role}"]
                lines.append(f"Exported: {datetime.now().isoformat()}")
                lines.append("=" * 50)
                
                for i, interaction in enumerate(history, 1):
                    lines.append(f"\nInteraction {i} ({interaction['timestamp']}):")
                    lines.append(f"Query: {interaction['query']}")
                    lines.append(f"Response: {interaction['response']}")
                    if interaction.get('sources'):
                        lines.append(f"Sources: {len(interaction['sources'])} documents")
                    lines.append("-" * 30)
                
                return "\n".join(lines)
            
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow(["timestamp", "query", "response", "source_count"])
                
                # Data
                for interaction in history:
                    writer.writerow([
                        interaction["timestamp"],
                        interaction["query"],
                        interaction["response"],
                        len(interaction.get("sources", []))
                    ])
                
                return output.getvalue()
            
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return f"Error exporting: {e}"

    def find_similar_queries(self, role: str, query: str, limit: int = 3) -> List[Dict]:
        """
        Find similar queries in conversation history.
        
        Args:
            role: User role
            query: Query to find similarities for
            limit: Maximum number of similar queries to return
            
        Returns:
            List of similar queries with metadata
        """
        try:
            history = self.conversations.get(role, [])
            
            if not history:
                return []
            
            # Simple similarity based on word overlap
            query_words = set(query.lower().split())
            similar_queries = []
            
            for interaction in history:
                hist_query = interaction["query"]
                hist_words = set(hist_query.lower().split())
                
                # Calculate Jaccard similarity
                intersection = query_words & hist_words
                union = query_words | hist_words
                
                if union:
                    similarity = len(intersection) / len(union)
                    
                    if similarity > 0.2:  # Threshold for similarity
                        similar_queries.append({
                            "query": hist_query,
                            "response": interaction["response"],
                            "timestamp": interaction["timestamp"],
                            "similarity": similarity
                        })
            
            # Sort by similarity and return top results
            similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_queries[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []

    def get_active_roles(self) -> List[str]:
        """Get list of roles with active conversations."""
        active_roles = []
        
        for role in self.conversations:
            if not self._is_session_expired(role) and self.conversations[role]:
                active_roles.append(role)
        
        return active_roles

    def get_conversation_topics(self, role: str) -> List[str]:
        """
        Extract main topics from conversation history.
        
        Args:
            role: User role
            
        Returns:
            List of identified topics
        """
        try:
            history = self.conversations.get(role, [])
            
            if not history:
                return []
            
            # Simple keyword extraction
            topics = set()
            
            for interaction in history:
                query = interaction["query"].lower()
                
                # Extract meaningful words (length > 3, not common words)
                words = query.split()
                common_words = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'and', 'or', 'but', 'with', 'for', 'can', 'you', 'please'}
                
                for word in words:
                    cleaned_word = word.strip('.,!?()[]{}";:')
                    if len(cleaned_word) > 3 and cleaned_word not in common_words:
                        topics.add(cleaned_word)
            
            return list(topics)[:10]  # Return top 10 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []

    def _update_session_metadata(self, role: str):
        """Update session metadata for a role."""
        try:
            if role not in self.session_metadata:
                self.session_metadata[role] = {
                    "session_start": datetime.now().isoformat(),
                    "interaction_count": 0
                }
            
            self.session_metadata[role]["last_activity"] = datetime.now().isoformat()
            self.session_metadata[role]["interaction_count"] += 1
            
        except Exception as e:
            logger.error(f"Error updating session metadata: {e}")

    def _is_session_expired(self, role: str) -> bool:
        """Check if a session has expired."""
        try:
            if role not in self.session_metadata:
                return False
            
            last_activity_str = self.session_metadata[role].get("last_activity")
            if not last_activity_str:
                return False
            
            last_activity = datetime.fromisoformat(last_activity_str)
            time_since_activity = datetime.now() - last_activity
            
            return time_since_activity.total_seconds() > self.conversation_timeout
            
        except Exception as e:
            logger.error(f"Error checking session expiry: {e}")
            return False

    def get_session_info(self, role: str) -> Dict[str, Any]:
        """
        Get session information for a role.
        
        Args:
            role: User role
            
        Returns:
            Session information dictionary
        """
        try:
            metadata = self.session_metadata.get(role, {})
            
            session_info = {
                "role": role,
                "is_active": not self._is_session_expired(role),
                "interaction_count": metadata.get("interaction_count", 0),
                "session_start": metadata.get("session_start"),
                "last_activity": metadata.get("last_activity"),
                "history_length": len(self.conversations.get(role, []))
            }
            
            # Calculate session duration if session exists
            if metadata.get("session_start") and metadata.get("last_activity"):
                start_time = datetime.fromisoformat(metadata["session_start"])
                last_time = datetime.fromisoformat(metadata["last_activity"])
                duration = last_time - start_time
                session_info["session_duration_minutes"] = duration.total_seconds() / 60
            
            return session_info
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {"error": str(e)}

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            expired_roles = []
            
            for role in list(self.conversations.keys()):
                if self._is_session_expired(role):
                    expired_roles.append(role)
            
            for role in expired_roles:
                self.clear_history(role)
                logger.info(f"Cleaned up expired session for role: {role}")
            
            return len(expired_roles)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
