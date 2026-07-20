"""
MCP Server Implementation for Final RAG Chatbot

This module implements the Model Context Protocol (MCP) server interface
for the Final RAG Chatbot system, providing standardized access to all
RAG functionality through the MCP protocol.
"""

import json
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Union

# Notification emitter signature: (method, params) -> None.
NotifyFn = Callable[[str, Dict[str, Any]], None]


def _format_progress_message(stage: str, current: Optional[int], total: Optional[int]) -> str:
    """Render a stage/current/total triple as a human-readable progress message."""
    if total and current is not None:
        return f"{stage}: {current}/{total}"
    if current is not None:
        return f"{stage}: {current}"
    return stage

# Add the parent directory to the Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our existing RAG system components
from core.final_rag_system import FinalRAGChatbot
from core.utils import setup_logging, validate_input

logger = logging.getLogger(__name__)


class MCPProtocolHandler:
    """
    Basic MCP Protocol Handler for the Final RAG Chatbot system.
    This implements the essential MCP protocol without external dependencies.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the MCP handler with RAG system."""
        self.config_path = config_path
        self.rag_system: Optional[FinalRAGChatbot] = None
        self.server_info = {
            "name": "Final RAG Chatbot",
            "version": "1.0.0"
        }
        
    async def handle_request(
        self,
        request: Dict[str, Any],
        notify: Optional[NotifyFn] = None,
    ) -> Dict[str, Any]:
        """Handle incoming MCP requests.

        ``notify`` is an optional callable used to emit JSON-RPC notifications
        (no ``id``) such as ``notifications/progress``. When the request params
        include ``_meta.progressToken``, long-running tools fan progress events
        through this callable.
        """
        request_id = request.get("id", "unknown")
        try:
            method = request.get("method", "")
            params = request.get("params", {})
            progress_token = (params.get("_meta") or {}).get("progressToken") if isinstance(params, dict) else None

            if method == "initialize":
                return await self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return await self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(request_id, params, notify=notify, progress_token=progress_token)
            elif method == "resources/list":
                return await self._handle_resources_list(request_id)
            elif method == "resources/read":
                return await self._handle_resource_read(request_id, params)
            elif method == "prompts/list":
                return await self._handle_prompts_list(request_id)
            elif method == "prompts/get":
                return await self._handle_prompt_get(request_id, params)
            else:
                return self._create_error_response(request_id, "Method not found", -32601)

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return self._create_error_response(request_id, str(e), -32603)

    def _build_progress_callback(
        self,
        notify: Optional[NotifyFn],
        progress_token: Optional[str],
    ) -> Optional[Callable[[str, Optional[int], Optional[int]], None]]:
        """Return a ``(stage, current, total)`` callback that fans events to MCP notifications."""
        if notify is None or progress_token is None:
            return None

        def _emit(stage: str, current: Optional[int], total: Optional[int]) -> None:
            payload: Dict[str, Any] = {
                "progressToken": progress_token,
                "progress": float(current) if current is not None else 0.0,
                "stage": stage,
                "message": _format_progress_message(stage, current, total),
            }
            if total is not None:
                payload["total"] = float(total)
            try:
                notify("notifications/progress", payload)
            except Exception:
                logger.debug("Progress notification emit failed", exc_info=True)

        return _emit
    
    async def _handle_initialize(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": self.server_info
            }
        }
    
    async def _handle_tools_list(self, request_id: str) -> Dict[str, Any]:
        """Handle tools listing request."""
        tools = [
            {
                "name": "chat",
                "description": "Chat with the RAG system using natural language queries",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user's natural language query"},
                        "role": {"type": "string", "default": "User", "enum": ["Admin", "Expert", "User", "Guest"]},
                        "stream": {"type": "boolean", "default": False}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "load_documents",
                "description": "Load documents into the RAG system for processing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Path to document(s) or directory"},
                        "role": {"type": "string", "default": "Admin", "enum": ["Admin", "Expert", "User", "Guest"]},
                        "document_type": {"type": "string", "default": "auto"}
                    },
                    "required": ["source"]
                }
            },
            {
                "name": "get_stats",
                "description": "Get system statistics and performance metrics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "default": "User", "enum": ["Admin", "Expert", "User", "Guest"]}
                    }
                }
            },
            {
                "name": "clear_conversation",
                "description": "Clear conversation history for a specific role",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "default": "User", "enum": ["Admin", "Expert", "User", "Guest"]}
                    }
                }
            },
            {
                "name": "clear_documents",
                "description": "Clear loaded documents, embeddings, and the persisted embeddings cache",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "default": "Admin", "enum": ["Admin", "Expert", "User", "Guest"]},
                        "delete_cache": {"type": "boolean", "default": True}
                    }
                }
            },
            {
                "name": "analyze_query",
                "description": "Analyze a query for intent, complexity, and suggested improvements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to analyze"},
                        "role": {"type": "string", "default": "User", "enum": ["Admin", "Expert", "User", "Guest"]}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_documents",
                "description": "Search for relevant documents based on a query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "role": {"type": "string", "default": "User", "enum": ["Admin", "Expert", "User", "Guest"]},
                        "max_results": {"type": "integer", "default": 5},
                        "threshold": {"type": "number", "default": 0.3}
                    },
                    "required": ["query"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools}
        }
    
    async def _handle_tool_call(
        self,
        request_id: str,
        params: Dict[str, Any],
        notify: Optional[NotifyFn] = None,
        progress_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        progress_callback = self._build_progress_callback(notify, progress_token)

        try:
            if tool_name == "chat":
                result = await self._tool_chat(arguments, progress_callback=progress_callback)
            elif tool_name == "load_documents":
                result = await self._tool_load_documents(arguments, progress_callback=progress_callback)
            elif tool_name == "get_stats":
                result = await self._tool_get_stats(arguments)
            elif tool_name == "clear_conversation":
                result = await self._tool_clear_conversation(arguments)
            elif tool_name == "clear_documents":
                result = await self._tool_clear_documents(arguments)
            elif tool_name == "analyze_query":
                result = await self._tool_analyze_query(arguments)
            elif tool_name == "search_documents":
                result = await self._tool_search_documents(arguments)
            else:
                return self._create_error_response(request_id, f"Unknown tool: {tool_name}", -32601)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return self._create_error_response(request_id, str(e), -32603)
    
    async def _tool_chat(
        self,
        arguments: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
    ) -> str:
        """Handle chat tool call."""
        query = arguments.get("query", "")
        role = arguments.get("role", "User")
        stream = arguments.get("stream", False)

        if not query:
            return "Error: Query parameter is required"

        rag_system = await self._get_rag_system(role)
        response = rag_system.chat(query, stream=False, progress_callback=progress_callback)

        if isinstance(response, dict):
            return json.dumps(response, indent=2)
        return str(response)

    async def _tool_load_documents(
        self,
        arguments: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
    ) -> str:
        """Handle load_documents tool call."""
        source = arguments.get("source", "")
        role = arguments.get("role", "Admin")
        document_type = arguments.get("document_type", "auto")

        if not source:
            return "Error: Source parameter is required"

        rag_system = await self._get_rag_system(role)
        success = rag_system.load_documents(source, document_type, progress_callback=progress_callback)

        if success:
            return rag_system.last_load_summary or f"Successfully loaded documents from: {source}"
        else:
            detail = f" {rag_system.last_error}" if getattr(rag_system, "last_error", "") else ""
            return f"Failed to load documents from: {source}.{detail}"
    
    async def _tool_get_stats(self, arguments: Dict[str, Any]) -> str:
        """Handle get_stats tool call."""
        role = arguments.get("role", "User")
        
        rag_system = await self._get_rag_system(role)
        stats = rag_system.get_stats()
        return json.dumps(stats, indent=2)
    
    async def _tool_clear_conversation(self, arguments: Dict[str, Any]) -> str:
        """Handle clear_conversation tool call."""
        role = arguments.get("role", "User")
        
        rag_system = await self._get_rag_system(role)
        rag_system.clear_conversation()
        return f"Conversation history cleared for role: {role}"

    async def _tool_clear_documents(self, arguments: Dict[str, Any]) -> str:
        """Handle clear_documents tool call."""
        role = arguments.get("role", "Admin")
        delete_cache = arguments.get("delete_cache", True)

        rag_system = await self._get_rag_system(role)
        result = rag_system.clear_documents(delete_cache=delete_cache)
        return json.dumps(result, indent=2)
    
    async def _tool_analyze_query(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_query tool call."""
        query = arguments.get("query", "")
        role = arguments.get("role", "User")
        
        if not query:
            return "Error: Query parameter is required"
        
        rag_system = await self._get_rag_system(role)
        analysis = rag_system.query_analyzer.analyze_query(query, role)
        return json.dumps(analysis, indent=2)
    
    async def _tool_search_documents(self, arguments: Dict[str, Any]) -> str:
        """Handle search_documents tool call."""
        query = arguments.get("query", "")
        role = arguments.get("role", "User")
        max_results = arguments.get("max_results", 5)
        threshold = arguments.get("threshold", 0.3)
        
        if not query:
            return "Error: Query parameter is required"
        
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
    
    async def _handle_resources_list(self, request_id: str) -> Dict[str, Any]:
        """Handle resources listing request."""
        resources = [
            {
                "uri": "rag://conversations/admin",
                "name": "Admin Conversation History",
                "description": "Access conversation history for Admin role",
                "mimeType": "application/json"
            },
            {
                "uri": "rag://conversations/user",
                "name": "User Conversation History", 
                "description": "Access conversation history for User role",
                "mimeType": "application/json"
            },
            {
                "uri": "rag://conversations/expert",
                "name": "Expert Conversation History",
                "description": "Access conversation history for Expert role",
                "mimeType": "application/json"
            },
            {
                "uri": "rag://conversations/guest",
                "name": "Guest Conversation History",
                "description": "Access conversation history for Guest role",
                "mimeType": "application/json"
            },
            {
                "uri": "rag://config",
                "name": "System Configuration",
                "description": "Access current system configuration",
                "mimeType": "application/json"
            },
            {
                "uri": "rag://documents/list",
                "name": "Document List",
                "description": "List all loaded documents",
                "mimeType": "application/json"
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": resources}
        }
    
    async def _handle_resource_read(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request."""
        uri = params.get("uri", "")
        
        try:
            if uri.startswith("rag://conversations/"):
                role = uri.split("/")[-1]
                rag_system = await self._get_rag_system(role)
                history = rag_system.conversation_manager.get_full_history(role)
                content = json.dumps(history, indent=2)
            elif uri == "rag://config":
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                content = json.dumps(config, indent=2)
            elif uri == "rag://documents/list":
                # Use Admin role to access all documents
                rag_system = await self._get_rag_system("Admin")
                doc_list = rag_system.get_document_summaries()
                content = json.dumps(doc_list, indent=2)
            else:
                return self._create_error_response(request_id, f"Unknown resource: {uri}", -32601)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": content
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return self._create_error_response(request_id, str(e), -32603)
    
    async def _handle_prompts_list(self, request_id: str) -> Dict[str, Any]:
        """Handle prompts listing request."""
        prompts = [
            {
                "name": "system_prompt",
                "description": "Get the system prompt for a specific role",
                "arguments": [
                    {
                        "name": "role",
                        "description": "User role (Admin, Expert, User, Guest)",
                        "required": False
                    }
                ]
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"prompts": prompts}
        }
    
    async def _handle_prompt_get(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt get request."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if name == "system_prompt":
            role = arguments.get("role", "User")
            rag_system = await self._get_rag_system(role)
            role_config = rag_system.config["roles"].get(role, rag_system.config["roles"]["User"])
            system_prompt = rag_system._create_system_prompt(role_config)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "description": f"System prompt for role: {role}",
                    "messages": [
                        {
                            "role": "system",
                            "content": {
                                "type": "text",
                                "text": system_prompt
                            }
                        }
                    ]
                }
            }
        else:
            return self._create_error_response(request_id, f"Unknown prompt: {name}", -32601)
    
    def _create_error_response(self, request_id: str, message: str, code: int) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
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


class MCPStdioServer:
    """
    STDIO-based MCP Server for the Final RAG Chatbot system.
    """

    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the STDIO server."""
        self.handler = MCPProtocolHandler(config_path)

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        """Emit a JSON-RPC notification (no ``id``) on stdout."""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            print(json.dumps(notification), flush=True)
        except Exception:
            logger.debug("Failed to write MCP notification", exc_info=True)

    async def run(self):
        """Run the STDIO server."""
        logger.info("Starting Final RAG Chatbot MCP Server (STDIO)")

        while True:
            try:
                # Read JSON-RPC request from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line.strip())

                # Handle the request - notify hook fans out progress events.
                response = await self.handler.handle_request(request, notify=self._notify)

                # Send response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Server error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": "Internal error"
                    }
                }
                print(json.dumps(error_response), flush=True)


async def main():
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final RAG Chatbot MCP Server")
    parser.add_argument("--config", default="config/config.json", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]  # Log to stderr to avoid interference with STDIO
    )
    
    # Create and run the server
    server = MCPStdioServer(args.config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
