"""
Advanced RAG Features Example

This example demonstrates advanced features of the Final RAG Chatbot:
- Conversation history management
- Role-based access control
- Performance monitoring
- Async operations
- Custom prompting strategies
"""

import sys
import asyncio
from datetime import datetime
from pathlib import Path

# Add src/ to the Python path for direct example execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.final_rag_system import FinalRAGChatbot
from examples.example_utils import ensure_sample_documents, response_to_text


def prepare_chatbot(role: str = "User") -> FinalRAGChatbot:
    """Create a chatbot and ensure the examples have documents to search."""
    chatbot = FinalRAGChatbot(role=role)
    if chatbot.embedding_manager.get_document_count() == 0:
        documents_dir = ensure_sample_documents()
        chatbot.load_documents(str(documents_dir))
    return chatbot

async def test_conversation_history():
    """Test conversation history management."""
    print(" Testing Conversation History")
    print("-" * 40)
    
    chatbot = prepare_chatbot("User")
    
    # Have a conversation
    conversation_flow = [
        "Hello, I'm interested in learning about Python",
        "What are the main advantages of Python?",
        "How does Python compare to Java?",
        "Can you summarize what we've discussed about Python?"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n[Turn {i}] User: {message}")
        response = response_to_text(await chatbot.chat_async(message))
        print(f"[Turn {i}] Bot: {response[:100]}...")
    
    # Show conversation history
    history = chatbot.conversation_manager.get_full_history(chatbot.role)
    print(f"\n Conversation Statistics:")
    print(f"   Total turns: {len(history)}")
    print(f"   Role: {chatbot.role}")
    
    # Export conversation
    export_file = f"conversation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    exported_data = chatbot.export_conversation("json")
    with open(export_file, 'w') as f:
        import json
        json.dump(exported_data, f, indent=2)
    print(f"   Exported to: {export_file}")

def test_role_based_access():
    """Test different roles and their capabilities."""
    print("\n Testing Role-Based Access Control")
    print("-" * 40)
    
    roles = ["User", "Expert", "Admin"]
    test_question = "Explain machine learning algorithms"
    
    for role in roles:
        print(f"\n Testing role: {role}")
        chatbot = prepare_chatbot(role)
        
        response = response_to_text(chatbot.chat(test_question, stream=False))
        print(f"   Response length: {len(response)} characters")
        print(f"   Response preview: {response[:80]}...")
        
        # Show role-specific features
        role_capabilities = {
            "User": ["basic_queries", "document_search"],
            "Expert": ["advanced_queries", "technical_analysis", "document_search"],
            "Admin": ["system_management", "user_management", "advanced_queries", "document_search"]
        }
        
        if role in role_capabilities:
            capabilities = role_capabilities[role]
            print(f"   Capabilities: {', '.join(capabilities)}")
            print(f"   Max context: {chatbot.config['system']['max_conversation_history']}")
        else:
            print(f"   Role capabilities: Unknown")

def test_performance_monitoring():
    """Test performance monitoring features."""
    print("\n Testing Performance Monitoring")
    print("-" * 40)
    
    chatbot = prepare_chatbot("User")
    
    # Perform several operations to generate metrics
    test_queries = [
        "What is artificial intelligence?",
        "Explain deep learning",
        "How does natural language processing work?",
        "What are the applications of machine learning?"
    ]
    
    print("Running test queries...")
    for query in test_queries:
        chatbot.chat(query, stream=False)
        print(f"    Processed: {query[:30]}...")
    
    # Get performance statistics
    stats = chatbot.get_stats()
    print(f"\n Performance Statistics:")
    for metric, value in stats.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")

async def test_async_operations():
    """Test asynchronous operations."""
    print("\n Testing Async Operations")
    print("-" * 40)
    
    chatbot = prepare_chatbot("User")
    
    # Prepare multiple queries
    queries = [
        "What is Python programming?",
        "Explain data structures",
        "How do algorithms work?",
        "What is software engineering?"
    ]
    
    print("Running concurrent queries...")
    start_time = datetime.now()
    
    # Run queries concurrently
    tasks = [chatbot.chat_async(query) for query in queries]
    responses = [response_to_text(response) for response in await asyncio.gather(*tasks)]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"    Processed {len(queries)} queries concurrently")
    print(f"   ⏱️  Total time: {duration:.2f} seconds")
    print(f"    Average time per query: {duration/len(queries):.2f} seconds")
    
    # Show response summaries
    for i, (query, response) in enumerate(zip(queries, responses), 1):
        print(f"   [{i}] {query[:25]}... → {len(response)} chars")

def test_custom_prompting():
    """Test custom prompting strategies."""
    print("\n Testing Custom Prompting Strategies")
    print("-" * 40)
    
    chatbot = prepare_chatbot("Expert")
    
    # Test different prompting approaches
    base_question = "Explain machine learning"
    
    prompt_strategies = [
        ("Basic", base_question),
        ("Detailed", f"Please provide a comprehensive explanation of {base_question.lower()} including examples"),
        ("Beginner", f"Explain {base_question.lower()} in simple terms for a beginner"),
        ("Technical", f"Provide a technical deep-dive into {base_question.lower()} with mathematical concepts"),
        ("Practical", f"How can I apply {base_question.lower()} in real-world projects?")
    ]
    
    for strategy_name, prompt in prompt_strategies:
        print(f"\n Strategy: {strategy_name}")
        response = response_to_text(chatbot.chat(prompt, stream=False))
        
        # Analyze response characteristics
        response_text = response if isinstance(response, str) else str(response)
        word_count = len(response_text.split())
        technical_terms = sum(1 for word in response_text.split() if any(term in word.lower() for term in ['algorithm', 'model', 'data', 'neural', 'regression', 'classification']))
        
        print(f"   Word count: {word_count}")
        print(f"   Technical terms: {technical_terms}")
        print(f"   Preview: {response[:80]}...")

def test_error_handling():
    """Test error handling and recovery."""
    print("\n️  Testing Error Handling")
    print("-" * 40)
    
    chatbot = prepare_chatbot("User")
    
    # Test various edge cases
    test_cases = [
        ("Empty query", ""),
        ("Very long query", "A" * 10000),
        ("Special characters", "What is  in programming?"),
        ("Multiple languages", "¿Qué es la programación? What is programming? 프로그래밍이란?"),
        ("Invalid characters", "What is \x00\x01\x02 programming?")
    ]
    
    for test_name, query in test_cases:
        print(f"\n Test: {test_name}")
        try:
            response = response_to_text(chatbot.chat(query, stream=False))
            print(f"    Handled successfully")
            print(f"   Response length: {len(response)}")
        except Exception as e:
            print(f"    Error: {e}")

def test_memory_usage():
    """Test memory usage and cleanup."""
    print("\n Testing Memory Management")
    print("-" * 40)
    
    import psutil
    import gc
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Create multiple chatbot instances
    chatbots = []
    for i in range(5):
        chatbot = prepare_chatbot("User")
        chatbot.chat(f"Test query {i}", stream=False)
        chatbots.append(chatbot)
    
    # Check memory after creating chatbots
    mid_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after 5 chatbots: {mid_memory:.2f} MB")
    print(f"Memory increase: {mid_memory - initial_memory:.2f} MB")
    
    # Clean up
    del chatbots
    gc.collect()
    
    # Check memory after cleanup
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after cleanup: {final_memory:.2f} MB")
    print(f"Memory recovered: {mid_memory - final_memory:.2f} MB")

async def main():
    """Main async function."""
    print(" Advanced RAG Features Example")
    print("=" * 50)
    
    try:
        # Test conversation history
        await test_conversation_history()
        
        # Test role-based access
        test_role_based_access()
        
        # Test performance monitoring
        test_performance_monitoring()
        
        # Test async operations
        await test_async_operations()
        
        # Test custom prompting
        test_custom_prompting()
        
        # Test error handling
        test_error_handling()
        
        # Test memory usage
        test_memory_usage()
        
        print("\n Advanced features example completed!")
        
    except Exception as e:
        print(f"\nError running advanced example: {e}")

def run_sync_only():
    """Run only synchronous tests."""
    print(" Advanced RAG Features Example (Sync Only)")
    print("=" * 50)
    
    try:
        # Test role-based access
        test_role_based_access()
        
        # Test performance monitoring
        test_performance_monitoring()
        
        # Test custom prompting
        test_custom_prompting()
        
        # Test error handling
        test_error_handling()
        
        # Test memory usage
        test_memory_usage()
        
        print("\n Synchronous advanced features example completed!")
        
    except Exception as e:
        print(f"\nError running example: {e}")

if __name__ == "__main__":
    try:
        # Try to run async version
        asyncio.run(main())
    except Exception as e:
        print(f"Async version failed: {e}")
        print("Running synchronous version...")
        run_sync_only()
