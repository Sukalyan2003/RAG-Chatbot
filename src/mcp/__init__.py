"""
MCP (Model Context Protocol) Integration for Final RAG Chatbot

This module provides MCP server and client implementations for the Final RAG
Chatbot system, enabling standardized access to all RAG functionality.

Components:
- MCPProtocolHandler: Core MCP protocol implementation
- MCPStdioServer: STDIO-based MCP server
- MCPClient: Client for communicating with MCP servers
- MCPIntegratedRAG: RAG system that uses MCP for communication
- FinalRAGChatbotMCP: Drop-in replacement for original FinalRAGChatbot

Usage:
    # As MCP Server (run from command line):
    python mcp_server.py --config config/config.json
    
    # As integrated RAG system:
    async with MCPIntegratedRAG(role="User") as rag:
        response = await rag.chat("Hello!")
        
    # As drop-in replacement:
    rag = FinalRAGChatbotMCP(role="User")
    # Note: Async operations recommended for MCP version
"""

from .mcp_server import MCPProtocolHandler, MCPStdioServer
from .client import MCPClient, MCPIntegratedRAG, FinalRAGChatbotMCP

__all__ = [
    "MCPProtocolHandler",
    "MCPStdioServer", 
    "MCPClient",
    "MCPIntegratedRAG",
    "FinalRAGChatbotMCP"
]

__version__ = "1.0.0"
