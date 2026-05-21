"""
MCP Server Implementation for Final RAG Chatbot

This module implements the Model Context Protocol (MCP) server interface
for the Final RAG Chatbot system, providing standardized access to all
RAG functionality through the MCP protocol.
"""

import json
import asyncio
import logging
from typing import Optional
import sys
from pathlib import Path

# Add the parent directory to the Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP packages not available. Installing them with: pip install mcp")

# Import our existing RAG system components
from core.final_rag_system import FinalRAGChatbot

logger = logging.getLogger(__name__)

class RAGMCPServer:
    """
    MCP Server wrapper for the Final RAG Chatbot system.
    Implements the Model Context Protocol to provide standardized access
    to all RAG functionality.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the MCP server with RAG system."""
        if not MCP_AVAILABLE:
            raise ImportError("Install mcp-requirements.txt to use the official MCP server")
        self.config_path = config_path
        self.rag_system: Optional[FinalRAGChatbot] = None
        self.server = FastMCP("Final RAG Chatbot")
        self.setup_mcp_handlers()
        
    def setup_mcp_handlers(self):
        """Set up MCP protocol handlers."""
        
        # Tool definitions
        @self.server.tool(
            name="chat",
            description="Chat with the RAG system using natural language queries"
        )
        async def chat_tool(
            query: str,
            role: str = "User",
            stream: bool = False
        ) -> str:
            """
            Process a chat query through the RAG system.
            
            Args:
                query: The user's natural language query
                role: User role (Admin, User, Guest)
                stream: Whether to use streaming response
            
            Returns:
                The RAG system's response to the query
            """
            try:
                rag_system = await self._get_rag_system(role)
                if stream:
                    # For MCP, we'll return the final response even if streaming is requested
                    # since MCP doesn't natively support streaming in tool responses
                    response = rag_system.chat(query, stream=False)
                else:
                    response = rag_system.chat(query, stream=False)
                
                return response
            except Exception as e:
                logger.error(f"Error in chat tool: {e}")
                return f"Error processing query: {str(e)}"
        
        @self.server.tool(
            name="load_documents",
            description="Load documents into the RAG system for processing"
        )
        async def load_documents_tool(
            source: str,
            role: str = "Admin",
            document_type: str = "auto"
        ) -> str:
            """
            Load documents into the RAG system.
            
            Args:
                source: Path to document(s) or directory
                role: User role (must have appropriate permissions)
                document_type: Type of documents to load
            
            Returns:
                Status message about document loading
            """
            try:
                rag_system = await self._get_rag_system(role)
                success = rag_system.load_documents(source, document_type)
                
                if success:
                    return rag_system.last_load_summary or f"Successfully loaded documents from: {source}"
                else:
                    detail = f" {rag_system.last_error}" if getattr(rag_system, "last_error", "") else ""
                    return f"Failed to load documents from: {source}.{detail}"
            except Exception as e:
                logger.error(f"Error loading documents: {e}")
                return f"Error loading documents: {str(e)}"
        
        @self.server.tool(
            name="get_stats",
            description="Get system statistics and performance metrics"
        )
        async def get_stats_tool(role: str = "User") -> str:
            """
            Get system statistics and performance metrics.
            
            Args:
                role: User role for accessing stats
            
            Returns:
                JSON string with system statistics
            """
            try:
                rag_system = await self._get_rag_system(role)
                stats = rag_system.get_stats()
                return json.dumps(stats, indent=2)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return f"Error getting stats: {str(e)}"
        
        @self.server.tool(
            name="clear_conversation",
            description="Clear conversation history for a specific role"
        )
        async def clear_conversation_tool(role: str = "User") -> str:
            """
            Clear conversation history for the specified role.
            
            Args:
                role: User role whose conversation to clear
            
            Returns:
                Confirmation message
            """
            try:
                rag_system = await self._get_rag_system(role)
                rag_system.clear_conversation()
                return f"Conversation history cleared for role: {role}"
            except Exception as e:
                logger.error(f"Error clearing conversation: {e}")
                return f"Error clearing conversation: {str(e)}"

        @self.server.tool(
            name="clear_documents",
            description="Clear loaded documents, embeddings, and the persisted embeddings cache"
        )
        async def clear_documents_tool(
            role: str = "Admin",
            delete_cache: bool = True
        ) -> str:
            """Clear loaded documents and embeddings."""
            try:
                rag_system = await self._get_rag_system(role)
                result = rag_system.clear_documents(delete_cache=delete_cache)
                return json.dumps(result, indent=2)
            except Exception as e:
                logger.error(f"Error clearing documents: {e}")
                return f"Error clearing documents: {str(e)}"
        
        @self.server.tool(
            name="analyze_query",
            description="Analyze a query for intent, complexity, and suggested improvements"
        )
        async def analyze_query_tool(
            query: str,
            role: str = "User"
        ) -> str:
            """
            Analyze a query for intent, complexity, and improvements.
            
            Args:
                query: The query to analyze
                role: User role for analysis context
            
            Returns:
                JSON string with query analysis results
            """
            try:
                rag_system = await self._get_rag_system(role)
                analysis = rag_system.query_analyzer.analyze_query(query, role)
                return json.dumps(analysis, indent=2)
            except Exception as e:
                logger.error(f"Error analyzing query: {e}")
                return f"Error analyzing query: {str(e)}"
        
        @self.server.tool(
            name="search_documents",
            description="Search for relevant documents based on a query"
        )
        async def search_documents_tool(
            query: str,
            role: str = "User",
            max_results: int = 5,
            threshold: float = 0.3
        ) -> str:
            """
            Search for relevant documents based on a query.
            
            Args:
                query: Search query
                role: User role for permissions
                max_results: Maximum number of results to return
                threshold: Similarity threshold for results
            
            Returns:
                JSON string with search results
            """
            try:
                rag_system = await self._get_rag_system(role)
                results = rag_system.embedding_manager.retrieve_documents(
                    query, max_results, threshold
                )
                
                # Format results for JSON serialization
                formatted_results = []
                for doc in results:
                    formatted_results.append({
                        "content": doc.get("content", "")[:500] + "...",  # Truncate for readability
                        "source": doc.get("metadata", {}).get("source", "Unknown"),
                        "score": doc.get("similarity_score", 0.0),
                        "metadata": doc.get("metadata", {})
                    })
                
                return json.dumps(formatted_results, indent=2)
            except Exception as e:
                logger.error(f"Error searching documents: {e}")
                return f"Error searching documents: {str(e)}"
        
        # Resource handlers
        @self.server.resource(
            uri="rag://conversations/{role}",
            name="Conversation History",
            description="Access conversation history for a specific role"
        )
        async def get_conversation_history(uri: str) -> str:
            """Get conversation history for a role."""
            try:
                # Extract role from URI
                role = uri.split("/")[-1]
                rag_system = await self._get_rag_system(role)
                history = rag_system.conversation_manager.get_full_history(role)
                return json.dumps(history, indent=2)
            except Exception as e:
                logger.error(f"Error getting conversation history: {e}")
                return f"Error getting conversation history: {str(e)}"
        
        @self.server.resource(
            uri="rag://config",
            name="System Configuration",
            description="Access current system configuration"
        )
        async def get_config() -> str:
            """Get current system configuration."""
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return json.dumps(config, indent=2)
            except Exception as e:
                logger.error(f"Error getting config: {e}")
                return f"Error getting config: {str(e)}"
        
        @self.server.resource(
            uri="rag://documents/list",
            name="Document List",
            description="List all loaded documents"
        )
        async def get_document_list() -> str:
            """Get list of all loaded documents."""
            try:
                # Use Admin role to access all documents
                rag_system = await self._get_rag_system("Admin")
                doc_list = rag_system.get_document_summaries()
                return json.dumps(doc_list, indent=2)
            except Exception as e:
                logger.error(f"Error getting document list: {e}")
                return f"Error getting document list: {str(e)}"
        
        # Prompt templates
        @self.server.prompt(
            name="system_prompt",
            description="Get the system prompt for a specific role"
        )
        async def get_system_prompt(role: str = "User") -> str:
            """Get the system prompt for a specific role."""
            try:
                rag_system = await self._get_rag_system(role)
                role_config = rag_system.config["roles"].get(role, rag_system.config["roles"]["User"])
                system_prompt = rag_system._create_system_prompt(role_config)
                return system_prompt
            except Exception as e:
                logger.error(f"Error getting system prompt: {e}")
                return f"Error getting system prompt: {str(e)}"
    
    async def _get_rag_system(self, role: str) -> FinalRAGChatbot:
        """Get the shared RAG system and switch its active role."""
        role = self._normalize_role(role)
        if self.rag_system is None:
            self.rag_system = FinalRAGChatbot(
                role=role,
                config_path=self.config_path
            )
        else:
            self.rag_system.role = role
        return self.rag_system

    def _normalize_role(self, role: str) -> str:
        """Normalize role names from MCP resource URIs and tool arguments."""
        for valid_role in ("Admin", "Expert", "User", "Guest"):
            if role.lower() == valid_role.lower():
                return valid_role
        return "User"
    
    async def run(self, transport: str = "stdio"):
        """Run the MCP server."""
        if transport == "stdio":
            await self.server.run_stdio()
            return
        raise ValueError(f"Unsupported transport: {transport}")


async def main():
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final RAG Chatbot MCP Server")
    parser.add_argument("--config", default="config/config.json", help="Configuration file path")
    parser.add_argument("--transport", default="stdio", choices=["stdio"], help="Transport method")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the server
    server = RAGMCPServer(args.config)
    await server.run(args.transport)


if __name__ == "__main__":
    asyncio.run(main())
