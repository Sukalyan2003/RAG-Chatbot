"""
Setup Script for Final RAG Chatbot

This script sets up the environment and initializes the RAG system.
"""

import os
import sys
import json
import logging
from pathlib import Path

def setup_directories(config):
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        config["paths"]["data_dir"],
        config["paths"]["documents_dir"],
        config["paths"]["embeddings_dir"],
        config["paths"]["logs_dir"],
        config["paths"]["cache_dir"]
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created: {directory}")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "sklearn", 
        "numpy",
        "requests",
        "beautifulsoup4",
        "pdfminer.six",
        "loguru",
        "rich"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_").replace(".", "_"))
            print(f"   ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("   ✅ All dependencies installed!")
    return True

def initialize_embedding_model(config):
    """Initialize and cache the embedding model."""
    print("🤖 Initializing embedding model...")
    
    try:
        provider = config["embedding"].get("provider", "sentence_transformers")
        model_name = config["embedding"]["model"]

        if provider == "ollama":
            import requests

            base_url = config["embedding"].get("base_url", config["llm"]["base_url"]).rstrip("/")
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model_name, "input": ["This is a test sentence."]},
                timeout=config["embedding"].get("timeout", 30),
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            if not embeddings:
                raise RuntimeError("Ollama returned no embeddings")
            print(f"   ✓ Connected to Ollama embedding model: {model_name}")
            print(f"   ✓ Embedding dimension: {len(embeddings[0])}")
        else:
            from sentence_transformers import SentenceTransformer

            print(f"   Loading model: {model_name}")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(["This is a test sentence."])
            print(f"   ✓ Model loaded successfully")
            print(f"   ✓ Embedding dimension: {embeddings.shape[1]}")

        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def test_llm_connection(config):
    """Test connection to LLM."""
    print("🔗 Testing LLM connection...")
    
    try:
        if config["llm"].get("provider") == "ollama":
            import requests

            response = requests.post(
                f"{config['llm']['base_url'].rstrip('/')}/api/chat",
                json={
                    "model": config["llm"]["model"],
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                    "options": {"num_predict": 10},
                },
                timeout=10,
            )
            response.raise_for_status()
        else:
            from openai import OpenAI

            client = OpenAI(
                base_url=config["llm"]["base_url"],
                api_key=config["llm"]["api_key"]
            )
            client.chat.completions.create(
                model=config["llm"]["model"],
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=10
            )
        
        print(f"   ✓ Connected to {config['llm']['provider']} LLM")
        print(f"   ✓ Model: {config['llm']['model']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM connection failed: {e}")
        print("   💡 Make sure your LLM server is running")
        return False

def create_sample_data():
    """Create sample documents for testing."""
    print("📄 Creating sample data...")
    
    sample_docs = {
        "sample_doc_1.txt": """
# Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

## Key Areas of AI:
- Machine Learning
- Natural Language Processing
- Computer Vision
- Robotics
- Expert Systems

AI has applications in various fields including healthcare, finance, transportation, and entertainment.
        """,
        
        "sample_doc_2.txt": """
# Machine Learning Fundamentals

Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

## Types of Machine Learning:
1. Supervised Learning - Learning with labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data  
3. Reinforcement Learning - Learning through trial and error

Common algorithms include linear regression, decision trees, neural networks, and support vector machines.
        """,
        
        "sample_doc_3.txt": """
# Natural Language Processing

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.

## NLP Applications:
- Text classification
- Sentiment analysis
- Language translation
- Chatbots and virtual assistants
- Information extraction

NLP combines computational linguistics with statistical, machine learning, and deep learning models.
        """
    }
    
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in sample_docs.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"   ✓ Created: {filename}")

def main():
    """Main setup function."""
    print("🚀 Final RAG Chatbot Setup")
    print("=" * 40)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        with open("config/config.json", 'r') as f:
            config = json.load(f)
        print("   ✓ Configuration loaded")
        
        # Setup directories
        setup_directories(config)
        
        # Check dependencies
        if not check_dependencies():
            print("\n❌ Setup failed due to missing dependencies")
            return False
        
        # Initialize embedding model
        if not initialize_embedding_model(config):
            print("\n⚠️  Embedding model initialization failed")
            print("The system will still work but performance may be affected")
        
        # Test LLM connection (optional)
        if not test_llm_connection(config):
            print("\n⚠️  LLM connection failed")
            print("You can still use the system with document processing features")
        
        # Create sample data
        create_sample_data()
        
        print("\n✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Start your LLM server (if using local)")
        print("2. Run: python -m src.core.final_rag_system --interactive")
        print("3. Or use: python -m src.core.final_rag_system --documents data/documents --interactive")
        
        return True
        
    except FileNotFoundError:
        print("❌ config/config.json not found. Please ensure the configuration file exists.")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
