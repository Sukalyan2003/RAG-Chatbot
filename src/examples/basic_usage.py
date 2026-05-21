"""
Basic Usage Example for Final RAG Chatbot

This example shows how to use the basic features of the RAG chatbot system.
"""

import sys
from pathlib import Path

# Add src/ to the Python path for direct example execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.final_rag_system import FinalRAGChatbot
from examples.example_utils import ensure_sample_documents, response_to_text

def basic_example():
    """Basic usage example."""
    print("🤖 Basic RAG Chatbot Example")
    print("-" * 40)
    
    try:
        # Initialize the chatbot
        print("Initializing chatbot...")
        chatbot = FinalRAGChatbot(role="User")
        
        # Load sample documents
        print("Loading sample documents...")
        documents_dir = ensure_sample_documents()
        success = chatbot.load_documents(str(documents_dir))
        
        if not success:
            print(f"Failed to load documents: {chatbot.last_error}")
            return
        
        print("Documents loaded successfully!")
        
        # Ask some sample questions
        sample_questions = [
            "What is artificial intelligence?",
            "Explain machine learning types",
            "What are NLP applications?",
            "Compare supervised and unsupervised learning"
        ]
        
        print("\n📝 Asking sample questions...")
        print("=" * 50)
        
        for question in sample_questions:
            print(f"\n❓ Question: {question}")
            response = response_to_text(chatbot.chat(question, stream=False))
            print(f"🤖 Answer: {response}")
            print("-" * 30)
        
        # Show statistics
        print("\n📊 System Statistics:")
        stats = chatbot.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run setup.py first and check your configuration.")

def interactive_example():
    """Interactive usage example."""
    print("🎮 Interactive RAG Chatbot Example")
    print("-" * 40)
    
    try:
        with FinalRAGChatbot(role="User") as chatbot:
            # Load documents
            print("Loading documents...")
            documents_dir = ensure_sample_documents()
            if not chatbot.load_documents(str(documents_dir)):
                print(f"Failed to load documents: {chatbot.last_error}")
                return
            
            print("Documents loaded! Type 'quit' to exit.")
            print("-" * 40)
            
            while True:
                try:
                    question = input("\n❓ Your question: ").strip()
                    
                    if question.lower() in ['quit', 'exit', 'bye']:
                        break
                    
                    if not question:
                        print("Please enter a question.")
                        continue
                    
                    response = response_to_text(chatbot.chat(question, stream=False))
                    print(f"🤖 Answer: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        print("Thanks for using the RAG chatbot!")
        
    except Exception as e:
        print(f"Error initializing chatbot: {e}")

def role_based_example():
    """Example showing role-based access."""
    print("👥 Role-based Access Example")
    print("-" * 40)
    
    roles = ["Guest", "User", "Admin"]
    
    for role in roles:
        print(f"\n🎭 Testing as {role}:")
        
        try:
            with FinalRAGChatbot(role=role) as chatbot:
                documents_dir = ensure_sample_documents()
                if not chatbot.load_documents(str(documents_dir)):
                    print(f"Failed to load documents: {chatbot.last_error}")
                    continue
                
                # Ask the same question with different roles
                question = "What is machine learning?"
                response = response_to_text(chatbot.chat(question, stream=False))
                
                print(f"   Question: {question}")
                print(f"   Response length: {len(response)} characters")
                print(f"   Response: {response[:100]}...")
                
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    print("Final RAG Chatbot Examples")
    print("=" * 50)
    
    print("\nChoose an example:")
    print("1. Basic Example")
    print("2. Interactive Example")  
    print("3. Role-based Example")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            basic_example()
        elif choice == "2":
            interactive_example()
        elif choice == "3":
            role_based_example()
        else:
            print("Invalid choice. Running basic example...")
            basic_example()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
