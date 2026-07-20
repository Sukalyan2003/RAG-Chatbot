"""
MCP Integration Examples for Final RAG Chatbot

This script demonstrates all the MCP features and capabilities
of the Final RAG Chatbot system.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

# Add src/ to the Python path for direct example execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.client import MCPIntegratedRAG, MCPClient
from core.utils import setup_logging

async def example_basic_chat():
    """Example: Basic chat functionality through MCP."""
    print("\n" + "="*50)
    print("EXAMPLE: Basic Chat Functionality")
    print("="*50)
    
    async with MCPIntegratedRAG(role="User") as rag:
        queries = [
            "Hello! How are you?",
            "What is machine learning?",
            "Explain neural networks in simple terms",
            "What are the main types of AI?"
        ]
        
        for query in queries:
            print(f"\n User: {query}")
            response = await rag.chat(query)
            print(f" Bot: {response}")

async def example_role_based_access():
    """Example: Role-based access control."""
    print("\n" + "="*50)
    print("EXAMPLE: Role-Based Access Control")
    print("="*50)
    
    roles = ["Guest", "User", "Admin"]
    query = "What can you tell me about system configuration?"
    
    for role in roles:
        print(f"\n--- Testing with role: {role} ---")
        async with MCPIntegratedRAG(role=role) as rag:
            response = await rag.chat(query)
            print(f" {role} Response: {response[:200]}...")

async def example_document_management():
    """Example: Document loading and management."""
    print("\n" + "="*50)
    print("EXAMPLE: Document Management")
    print("="*50)
    
    async with MCPIntegratedRAG(role="Admin") as rag:
        # Try to load documents (this will depend on what's available)
        test_paths = [
            "docs/README.md",
            "docs/QUICKSTART.md",
            "config/config.json"
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                print(f"\n Loading document: {path}")
                success = await rag.load_documents(path)
                print(f" Success: {success}")
                break
        else:
            print(" No test documents found to load")

async def example_query_analysis():
    """Example: Query analysis and understanding."""
    print("\n" + "="*50)
    print("EXAMPLE: Query Analysis")
    print("="*50)
    
    async with MCPIntegratedRAG(role="User") as rag:
        queries = [
            "What is the meaning of life?",
            "How do I install Python packages?",
            "Explain quantum computing",
            "What are the benefits of using RAG systems?"
        ]
        
        for query in queries:
            print(f"\n Analyzing: {query}")
            analysis = await rag.analyze_query(query)
            print(f" Analysis: {json.dumps(analysis, indent=2)}")

async def example_document_search():
    """Example: Document search and retrieval."""
    print("\n" + "="*50)
    print("EXAMPLE: Document Search")
    print("="*50)
    
    async with MCPIntegratedRAG(role="User") as rag:
        search_queries = [
            "machine learning",
            "configuration",
            "installation",
            "python"
        ]
        
        for search_query in search_queries:
            print(f"\n Searching for: {search_query}")
            results = await rag.search_documents(search_query, max_results=3)
            print(f" Found {len(results)} documents:")
            
            for i, doc in enumerate(results, 1):
                print(f"  {i}. {doc.get('source', 'Unknown')} (Score: {doc.get('score', 0):.3f})")
                print(f"     Preview: {doc.get('content', '')[:100]}...")

async def example_system_monitoring():
    """Example: System statistics and monitoring."""
    print("\n" + "="*50)
    print("EXAMPLE: System Monitoring")
    print("="*50)
    
    async with MCPIntegratedRAG(role="Admin") as rag:
        # Chat a bit to generate some stats
        await rag.chat("Hello")
        await rag.chat("How are you?")
        await rag.chat("What can you do?")
        
        # Get statistics
        stats = await rag.get_stats()
        print(" System Statistics:")
        print(json.dumps(stats, indent=2))

async def example_conversation_management():
    """Example: Conversation history management."""
    print("\n" + "="*50)
    print("EXAMPLE: Conversation Management")
    print("="*50)
    
    async with MCPIntegratedRAG(role="User") as rag:
        # Have a conversation
        conversation = [
            "Hi there!",
            "What's your name?",
            "What can you help me with?",
            "Tell me about AI"
        ]
        
        print(" Having a conversation...")
        for message in conversation:
            print(f" User: {message}")
            response = await rag.chat(message)
            print(f" Bot: {response[:100]}...")
        
        print("\n Clearing conversation history...")
        await rag.clear_conversation()
        print(" Conversation cleared!")

async def example_mcp_client_direct():
    """Example: Direct MCP client usage."""
    print("\n" + "="*50)
    print("EXAMPLE: Direct MCP Client Usage")
    print("="*50)
    
    # This example shows how to use the MCP client directly
    server_script = str(Path(__file__).parent.parent / "src" / "mcp" / "mcp_server.py")
    
    client = MCPClient([
        sys.executable, server_script,
        "--config", "config/config.json"
    ])
    
    try:
        await client.start_server()
        print(" MCP Server started successfully")
        
        # Test basic chat
        response = await client.chat("Hello from direct MCP client!")
        print(f" Direct MCP Response: {response}")
        
        # Test getting configuration
        config = await client.get_config()
        print(f"️  MCP Config Keys: {list(config.keys())}")
        
    except Exception as e:
        print(f" Error with direct MCP client: {e}")
    finally:
        await client.stop_server()
        print(" MCP Server stopped")

async def example_error_handling():
    """Example: Error handling and recovery."""
    print("\n" + "="*50)
    print("EXAMPLE: Error Handling")
    print("="*50)
    
    async with MCPIntegratedRAG(role="User") as rag:
        # Test various error conditions
        error_tests = [
            ("", "Empty query"),
            ("x" * 2000, "Very long query"),
            ("SELECT * FROM users;", "SQL injection attempt"),
            ("<script>alert('xss')</script>", "XSS attempt")
        ]
        
        for test_input, description in error_tests:
            print(f"\n Testing: {description}")
            try:
                response = await rag.chat(test_input)
                print(f" Handled gracefully: {response[:100]}...")
            except Exception as e:
                print(f" Error: {e}")

async def run_all_examples():
    """Run all MCP examples."""
    print(" Starting Final RAG Chatbot MCP Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Chat", example_basic_chat),
        ("Role-Based Access", example_role_based_access),
        ("Document Management", example_document_management),
        ("Query Analysis", example_query_analysis),
        ("Document Search", example_document_search),
        ("System Monitoring", example_system_monitoring),
        ("Conversation Management", example_conversation_management),
        ("Direct MCP Client", example_mcp_client_direct),
        ("Error Handling", example_error_handling)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n Running example: {name}")
            await example_func()
            print(f" Completed: {name}")
        except Exception as e:
            print(f" Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(" All MCP examples completed!")

async def interactive_demo():
    """Interactive demonstration of MCP features."""
    print("\n Interactive MCP Demo")
    print("Type a message to chat, or 'help' for commands. Type 'quit' to exit.")
    print("-" * 40)
    
    async with MCPIntegratedRAG(role="User") as rag:
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'help':
                    print("""
Available commands:
- <message>          : Chat with the system
- chat <message>     : Chat with the system
- stats              : Show system statistics
- analyze <query>    : Analyze a query
- search <terms>     : Search documents
- load <path>        : Load a document or directory (uses Admin role)
- docs               : List loaded document chunks
- clear              : Clear conversation
- reset              : Clear loaded documents and embeddings cache
- role <role>        : Change role (Guest/User/Expert/Admin)
- help               : Show this help
- quit               : Exit
                    """)
                elif command.startswith('chat '):
                    message = command[5:]
                    response = await rag.chat(message)
                    print(f" {response}")
                elif command == 'stats':
                    stats = await rag.get_stats()
                    print(f" {json.dumps(stats, indent=2)}")
                elif command.startswith('analyze '):
                    query = command[8:]
                    analysis = await rag.analyze_query(query)
                    print(f" {json.dumps(analysis, indent=2)}")
                elif command.startswith('search '):
                    terms = command[7:]
                    results = await rag.search_documents(terms)
                    print(f" Found {len(results)} documents")
                    for i, doc in enumerate(results[:3], 1):
                        print(f"  {i}. {doc.get('source', 'Unknown')}")
                elif command == 'load' or command.startswith('load '):
                    path = command[5:].strip() if command.startswith('load ') else ""
                    if not path:
                        print(" Usage: load <path>")
                    else:
                        previous_role = rag.role
                        rag.role = "Admin"
                        success = await rag.load_documents(path)
                        rag.role = previous_role
                        status = "" if success else ""
                        print(f"{status} {rag.last_operation_message}")
                elif command == 'docs':
                    documents = await rag.get_document_list()
                    print(f" Loaded documents: {len(documents)}")
                    for i, doc in enumerate(documents[:10], 1):
                        chunks = doc.get('chunks', 0)
                        chunk_label = "chunk" if chunks == 1 else "chunks"
                        print(
                            f"  {i}. {doc.get('document', doc.get('source', 'Unknown'))} "
                            f"({doc.get('type', 'unknown')}, {chunks} {chunk_label})"
                        )
                    if len(documents) > 10:
                        print(f"  ... and {len(documents) - 10} more documents")
                elif command == 'clear':
                    await rag.clear_conversation()
                    print(" Conversation cleared")
                elif command == 'reset':
                    message = await rag.clear_documents(delete_cache=True)
                    print(f" Cleared documents/cache: {message}")
                elif command == 'role' or command.startswith('role '):
                    role = command[5:].strip().title() if command.startswith('role ') else ""
                    valid_roles = {"Guest", "User", "Expert", "Admin"}
                    if role in valid_roles:
                        rag.role = role
                        print(f" Role changed to {role}")
                    else:
                        print(" Usage: role <Guest|User|Expert|Admin>")
                elif command:
                    response = await rag.chat(command)
                    print(f" {response}")
                else:
                    print(" Type a message, or 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final RAG Chatbot MCP Examples")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--example", help="Run specific example")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO", "data/logs")
    
    if args.interactive:
        asyncio.run(interactive_demo())
    elif args.example:
        # Map example names to functions
        examples = {
            "chat": example_basic_chat,
            "roles": example_role_based_access,
            "documents": example_document_management,
            "analysis": example_query_analysis,
            "search": example_document_search,
            "monitoring": example_system_monitoring,
            "conversation": example_conversation_management,
            "client": example_mcp_client_direct,
            "errors": example_error_handling
        }
        
        if args.example in examples:
            asyncio.run(examples[args.example]())
        else:
            print(f"Unknown example: {args.example}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        asyncio.run(run_all_examples())

if __name__ == "__main__":
    main()
