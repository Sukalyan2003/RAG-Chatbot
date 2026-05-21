"""
MCP Client Interface for Final RAG Chatbot

This module provides a client interface to interact with the MCP server
and integrates MCP functionality into the existing RAG system.
"""

import json
import asyncio
import logging
import subprocess
import sys
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client interface for communicating with the MCP server.
    """
    
    def __init__(self, server_command: List[str], config_path: str = "config/config.json"):
        """
        Initialize the MCP client.
        
        Args:
            server_command: Command to start the MCP server
            config_path: Path to configuration file
        """
        self.server_command = server_command
        self.config_path = config_path
        self.process = None
        self.request_id = 0
    
    async def start_server(self):
        """Start the MCP server process."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info("MCP server started successfully")
            
            # Initialize the server
            await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "Final RAG Chatbot Client",
                    "version": "1.0.0"
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                if self.process.returncode is None:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        self.process.kill()
                        await self.process.wait()
                logger.info("MCP server stopped")
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"Error stopping MCP server: {e!r}")
            finally:
                self.process = None
    
    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a request to the MCP server."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP server not started or pipes not available")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": str(self.request_id),
            "method": method,
            "params": params or {}
        }
        
        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()
        
        # Read response. The server should emit only JSON-RPC on stdout, but
        # tolerate accidental log lines so one noisy dependency does not break
        # the interactive client.
        response = None
        while True:
            response_line = await self.process.stdout.readline()
            if not response_line:
                stderr = ""
                if self.process.stderr:
                    try:
                        stderr_bytes = await asyncio.wait_for(self.process.stderr.read(), timeout=0.5)
                        stderr = stderr_bytes.decode(errors="replace").strip()
                    except asyncio.TimeoutError:
                        pass
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"MCP server closed stdout before responding{detail}")

            decoded = response_line.decode(errors="replace").strip()
            if not decoded:
                continue

            try:
                response = json.loads(decoded)
                break
            except json.JSONDecodeError:
                logger.warning(f"Ignoring non-JSON MCP stdout line: {decoded[:200]}")
                continue
        
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        
        return response.get("result", {})
    
    async def chat(self, query: str, role: str = "User", stream: bool = False) -> str:
        """Chat with the RAG system through MCP."""
        result = await self._send_request("tools/call", {
            "name": "chat",
            "arguments": {
                "query": query,
                "role": role,
                "stream": stream
            }
        })
        
        # Extract text from MCP response
        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return "No response received"
    
    async def load_documents(self, source: str, role: str = "Admin", document_type: str = "auto") -> str:
        """Load documents through MCP."""
        result = await self._send_request("tools/call", {
            "name": "load_documents", 
            "arguments": {
                "source": source,
                "role": role,
                "document_type": document_type
            }
        })
        
        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return "No response received"
    
    async def get_stats(self, role: str = "User") -> Dict[str, Any]:
        """Get system statistics through MCP."""
        result = await self._send_request("tools/call", {
            "name": "get_stats",
            "arguments": {
                "role": role
            }
        })
        
        content = result.get("content", [])
        if content and len(content) > 0:
            stats_text = content[0].get("text", "{}")
            return json.loads(stats_text)
        return {}
    
    async def clear_conversation(self, role: str = "User") -> str:
        """Clear conversation through MCP."""
        result = await self._send_request("tools/call", {
            "name": "clear_conversation",
            "arguments": {
                "role": role
            }
        })
        
        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return "No response received"

    async def clear_documents(self, role: str = "Admin", delete_cache: bool = True) -> str:
        """Clear loaded documents and embeddings through MCP."""
        result = await self._send_request("tools/call", {
            "name": "clear_documents",
            "arguments": {
                "role": role,
                "delete_cache": delete_cache,
            }
        })

        content = result.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return "No response received"
    
    async def analyze_query(self, query: str, role: str = "User") -> Dict[str, Any]:
        """Analyze query through MCP."""
        result = await self._send_request("tools/call", {
            "name": "analyze_query",
            "arguments": {
                "query": query,
                "role": role
            }
        })
        
        content = result.get("content", [])
        if content and len(content) > 0:
            analysis_text = content[0].get("text", "{}")
            return json.loads(analysis_text)
        return {}
    
    async def search_documents(self, query: str, role: str = "User", max_results: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search documents through MCP."""
        result = await self._send_request("tools/call", {
            "name": "search_documents",
            "arguments": {
                "query": query,
                "role": role,
                "max_results": max_results,
                "threshold": threshold
            }
        })
        
        content = result.get("content", [])
        if content and len(content) > 0:
            results_text = content[0].get("text", "[]")
            return json.loads(results_text)
        return []
    
    async def get_conversation_history(self, role: str) -> List[Dict[str, Any]]:
        """Get conversation history through MCP resources."""
        uri = f"rag://conversations/{role.lower()}"
        result = await self._send_request("resources/read", {
            "uri": uri
        })
        
        contents = result.get("contents", [])
        if contents and len(contents) > 0:
            history_text = contents[0].get("text", "[]")
            return json.loads(history_text)
        return []
    
    async def get_config(self) -> Dict[str, Any]:
        """Get system configuration through MCP resources."""
        result = await self._send_request("resources/read", {
            "uri": "rag://config"
        })
        
        contents = result.get("contents", [])
        if contents and len(contents) > 0:
            config_text = contents[0].get("text", "{}")
            return json.loads(config_text)
        return {}
    
    async def get_document_list(self) -> List[Dict[str, Any]]:
        """Get document list through MCP resources."""
        result = await self._send_request("resources/read", {
            "uri": "rag://documents/list"
        })
        
        contents = result.get("contents", [])
        if contents and len(contents) > 0:
            docs_text = contents[0].get("text", "[]")
            return json.loads(docs_text)
        return []
    
    async def get_system_prompt(self, role: str = "User") -> str:
        """Get system prompt through MCP prompts."""
        result = await self._send_request("prompts/get", {
            "name": "system_prompt",
            "arguments": {
                "role": role
            }
        })
        
        messages = result.get("messages", [])
        if messages and len(messages) > 0:
            content = messages[0].get("content", {})
            return content.get("text", "")
        return ""


class MCPIntegratedRAG:
    """
    RAG system that uses MCP for communication.
    This class provides the same interface as FinalRAGChatbot but uses MCP internally.
    """
    
    def __init__(self, role: str = "User", config_path: str = "config/config.json", use_mcp: bool = True):
        """
        Initialize the MCP-integrated RAG system.
        
        Args:
            role: User role
            config_path: Path to configuration file
            use_mcp: Whether to use MCP (if False, falls back to direct usage)
        """
        self.role = role
        self.config_path = config_path
        self.use_mcp = use_mcp
        self.client = None
        self.last_operation_message = ""
        
        if use_mcp:
            # Setup MCP client
            server_script = str(Path(__file__).parent / "mcp_server.py")
            self.client = MCPClient([
                sys.executable, server_script,
                "--config", config_path
            ], config_path)
        else:
            # Fallback to direct import
            from core.final_rag_system import FinalRAGChatbot
            self.direct_rag = FinalRAGChatbot(role=role, config_path=config_path)
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.use_mcp and self.client:
            await self.client.start_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.use_mcp and self.client:
            await self.client.stop_server()
    
    async def chat(self, query: str, stream: bool = False) -> str:
        """Chat with the system."""
        if self.use_mcp and self.client:
            return await self.client.chat(query, self.role, stream)
        else:
            self.direct_rag.role = self.role
            response = self.direct_rag.chat(query, stream)
            if isinstance(response, dict):
                return json.dumps(response, indent=2)
            return str(response)
    
    async def load_documents(self, source: str, document_type: str = "auto") -> bool:
        """Load documents."""
        if self.use_mcp and self.client:
            response = await self.client.load_documents(source, self.role, document_type)
            self.last_operation_message = response
            return not response.lower().startswith(("failed", "error"))
        else:
            self.direct_rag.role = self.role
            success = self.direct_rag.load_documents(source, document_type)
            self.last_operation_message = (
                self.direct_rag.last_load_summary or f"Successfully loaded documents from: {source}"
                if success
                else f"Failed to load documents from: {source}. {self.direct_rag.last_error}"
            )
            return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if self.use_mcp and self.client:
            return await self.client.get_stats(self.role)
        else:
            self.direct_rag.role = self.role
            return self.direct_rag.get_stats()
    
    async def clear_conversation(self):
        """Clear conversation history."""
        if self.use_mcp and self.client:
            await self.client.clear_conversation(self.role)
        else:
            self.direct_rag.role = self.role
            self.direct_rag.clear_conversation()

    async def clear_documents(self, delete_cache: bool = True) -> str:
        """Clear loaded documents and embeddings."""
        if self.use_mcp and self.client:
            self.last_operation_message = await self.client.clear_documents("Admin", delete_cache)
            return self.last_operation_message

        self.direct_rag.role = "Admin"
        result = self.direct_rag.clear_documents(delete_cache)
        self.last_operation_message = json.dumps(result, indent=2)
        return self.last_operation_message
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query."""
        if self.use_mcp and self.client:
            return await self.client.analyze_query(query, self.role)
        else:
            self.direct_rag.role = self.role
            # The analyze_query method may return a string, so we handle that
            result = self.direct_rag.query_analyzer.analyze_query(query, self.role)
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"analysis": result}
            return result
    
    async def search_documents(self, query: str, max_results: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search documents."""
        if self.use_mcp and self.client:
            return await self.client.search_documents(query, self.role, max_results, threshold)
        else:
            self.direct_rag.role = self.role
            return self.direct_rag.embedding_manager.retrieve_documents(query, max_results, threshold)

    async def get_document_list(self) -> List[Dict[str, Any]]:
        """Get loaded document list."""
        if self.use_mcp and self.client:
            return await self.client.get_document_list()

        return self.direct_rag.get_document_summaries()


# Compatibility wrapper for existing code
class FinalRAGChatbotMCP(MCPIntegratedRAG):
    """
    Drop-in replacement for FinalRAGChatbot that uses MCP.
    Provides the same interface but with MCP backend.
    """
    
    def __init__(self, role: str = "User", config_path: str = "config/config.json", custom_config: Optional[Dict] = None):
        """Initialize with the same signature as original FinalRAGChatbot."""
        super().__init__(role=role, config_path=config_path, use_mcp=True)
        self.custom_config = custom_config
    
    def __enter__(self):
        """Sync context manager - not recommended, use async version."""
        import warnings
        warnings.warn("Sync context manager not supported with MCP. Use async 'async with' instead.")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager - not recommended."""
        pass


async def test_mcp_integration():
    """Test the MCP integration."""
    async with MCPIntegratedRAG(role="User") as rag:
        # Test basic chat
        response = await rag.chat("Hello, how are you?")
        print(f"Chat response: {response}")
        
        # Test stats
        stats = await rag.get_stats()
        print(f"Stats: {stats}")
        
        # Test query analysis
        analysis = await rag.analyze_query("What is machine learning?")
        print(f"Query analysis: {analysis}")


if __name__ == "__main__":
    asyncio.run(test_mcp_integration())
