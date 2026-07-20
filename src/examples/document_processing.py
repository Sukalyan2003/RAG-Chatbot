"""
Document Processing Example

This example demonstrates the document processing capabilities
of the Final RAG Chatbot system.
"""

import sys
import os
from pathlib import Path

# Add src/ to the Python path for direct example execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.document_processor import DocumentProcessor
from core.final_rag_system import FinalRAGChatbot
from examples.example_utils import response_to_text

def create_test_documents():
    """Create various test documents for processing."""
    print(" Creating test documents...")
    
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Text document
    with open(test_dir / "sample.txt", "w", encoding="utf-8") as f:
        f.write("""
Python Programming Guide

Python is a high-level, interpreted programming language with dynamic semantics. 
Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
make it very attractive for Rapid Application Development.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Large community support
- Excellent for data science and AI

Python is widely used in web development, data analysis, artificial intelligence, 
scientific computing, and automation.
        """)
    
    # JSON document
    import json
    data = {
        "title": "Data Science Fundamentals",
        "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
        "topics": ["Statistics", "Machine Learning", "Data Visualization", "Programming"],
        "tools": ["Python", "R", "SQL", "Tableau", "Jupyter"]
    }
    
    with open(test_dir / "data_science.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    # CSV document
    with open(test_dir / "ml_algorithms.csv", "w", encoding="utf-8") as f:
        f.write("""Algorithm,Type,Use Case,Description
Linear Regression,Supervised,Prediction,Predicts continuous values based on linear relationships
Decision Trees,Supervised,Classification,Creates a tree-like model of decisions
K-Means,Unsupervised,Clustering,Groups data into k clusters
Neural Networks,Supervised/Unsupervised,Various,Mimics human brain structure for learning
Random Forest,Supervised,Classification/Regression,Ensemble method using multiple decision trees""")
    
    print(f"    Created test documents in {test_dir}")
    return test_dir

def test_document_processor():
    """Test the document processor with various file types."""
    print("\n Testing Document Processor")
    print("-" * 40)
    
    # Load configuration
    import json
    with open("config/config.json", "r") as f:
        config = json.load(f)
    
    processor = DocumentProcessor(config)
    
    # Create test documents
    test_dir = create_test_documents()
    
    try:
        # Process all documents in the test directory
        print(f"Processing documents from {test_dir}...")
        documents = processor.process_documents(str(test_dir))
        
        print(f"\n Processing Results:")
        print(f"   Total document chunks: {len(documents)}")
        
        # Show details for each document type
        doc_types = {}
        for doc in documents:
            doc_type = doc['metadata']['type']
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        for doc_type, docs in doc_types.items():
            print(f"\n    {doc_type.upper()} files:")
            print(f"      Chunks: {len(docs)}")
            
            # Show sample content
            if docs:
                sample_doc = docs[0]
                content_preview = sample_doc['content'][:200] + "..." if len(sample_doc['content']) > 200 else sample_doc['content']
                print(f"      Sample content: {content_preview}")
                print(f"      Metadata: {sample_doc['metadata']}")
        
        return documents
        
    except Exception as e:
        print(f"Error processing documents: {e}")
        return []

def test_with_chatbot():
    """Test processed documents with the chatbot."""
    print("\n Testing with Chatbot")
    print("-" * 40)
    
    try:
        # Initialize chatbot
        chatbot = FinalRAGChatbot(role="User")
        
        # Load the test documents
        test_dir = "test_documents"
        if not os.path.exists(test_dir):
            print("Test documents not found. Run test_document_processor first.")
            return
        
        print("Loading test documents...")
        success = chatbot.load_documents(test_dir)
        
        if not success:
            print("Failed to load documents")
            return
        
        print("Documents loaded successfully!")
        
        # Test questions related to the documents
        test_questions = [
            "What are the key features of Python?",
            "What is data science?",
            "Tell me about machine learning algorithms",
            "What is the difference between supervised and unsupervised learning?",
            "What tools are used in data science?"
        ]
        
        print("\n Testing questions:")
        for question in test_questions:
            print(f"\n   Q: {question}")
            response = response_to_text(chatbot.chat(question, stream=False))
            print(f"   A: {response[:150]}...")
        
        # Show statistics
        print(f"\nFinal Statistics:")
        stats = chatbot.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"Error testing with chatbot: {e}")

def test_web_processing():
    """Test web content processing (if available)."""
    print("\n Testing Web Content Processing")
    print("-" * 40)
    
    try:
        # Load configuration
        import json
        with open("config/config.json", "r") as f:
            config = json.load(f)
        
        processor = DocumentProcessor(config)
        
        # Test with a simple web page (if web scraping is available)
        test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        
        print(f"Attempting to process: {test_url}")
        documents = processor.process_documents(test_url, "web")
        
        if documents:
            print(f"    Successfully processed web content")
            print(f"    Generated {len(documents)} chunks")
            
            # Show sample content
            if documents:
                sample = documents[0]['content'][:200] + "..."
                print(f"   Sample: {sample}")
        else:
            print("    Web processing not available or failed")
            
    except Exception as e:
        print(f"    Web processing error: {e}")

def cleanup_test_files():
    """Clean up test files."""
    print("\n Cleaning up test files...")
    
    try:
        import shutil
        test_dir = Path("test_documents")
        
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("    Test files cleaned up")
        else:
            print("    No test files to clean up")
            
    except Exception as e:
        print(f"    Error cleaning up: {e}")

def main():
    """Main function."""
    print(" Document Processing Example")
    print("=" * 50)
    
    try:
        # Test document processor
        documents = test_document_processor()
        
        if documents:
            # Test with chatbot
            test_with_chatbot()
            
            # Test web processing
            test_web_processing()
        
        # Ask user if they want to keep test files
        keep_files = input("\nKeep test files? (y/n): ").strip().lower()
        if keep_files != 'y':
            cleanup_test_files()
        
        print("\n Document processing example completed!")
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
        cleanup_test_files()
    except Exception as e:
        print(f"\nError running example: {e}")

if __name__ == "__main__":
    main()
