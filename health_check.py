"""
System Health Check for Final RAG Chatbot MCP System

This script checks the installation and configuration of the
Final RAG Chatbot system with MCP integration.
"""

import sys
import os
import json
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check required dependencies."""
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("streamlit", "streamlit"),
        ("sklearn", "scikit-learn"),
        ("pdfminer", "pdfminer.six"),
        ("bs4", "beautifulsoup4"),
        ("requests", "requests"),
    ]
    
    optional_packages = [
        ("sentence_transformers", "sentence-transformers (local embedding fallback)"),
        ("openai", "openai (chat-completion compatible LLM fallback)"),
        ("docx", "python-docx"),
        ("mcp", "mcp (official MCP server implementation)"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("websockets", "websockets"),
        ("httpx", "httpx"),
        ("anyio", "anyio"),
    ]
    
    results = []
    all_available = True
    
    results.append("Required Dependencies:")
    for module_name, package_name in required_packages:
        if importlib.util.find_spec(module_name):
            results.append(f"✅ {package_name}")
        else:
            results.append(f"❌ {package_name} (pip install {package_name})")
            all_available = False
    
    results.append("\nOptional Dependencies (enhanced features):")
    for module_name, package_name in optional_packages:
        if importlib.util.find_spec(module_name):
            results.append(f"✅ {package_name}")
        else:
            results.append(f"⚠️  {package_name} (pip install {package_name})")
    
    return all_available, results

def check_configuration() -> Tuple[bool, List[str]]:
    """Check configuration files."""
    config_files = [
        "config/config.json",
        "requirements.txt",
        "mcp-requirements.txt"
    ]
    
    results = []
    all_exist = True
    
    for config_file in config_files:
        if os.path.exists(config_file):
            results.append(f"✅ {config_file}")
            
            # Validate JSON config
            if config_file.endswith('.json'):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Check MCP configuration
                    if 'mcp' in config:
                        results.append(f"  ✅ MCP configuration found")
                    else:
                        results.append(f"  ⚠️  MCP configuration missing (will use defaults)")
                        
                except json.JSONDecodeError as e:
                    results.append(f"  ❌ Invalid JSON: {e}")
                    all_exist = False
        else:
            results.append(f"❌ {config_file}")
            all_exist = False
    
    return all_exist, results

def check_core_modules() -> Tuple[bool, List[str]]:
    """Check core system modules."""
    sys.path.append(str(Path(__file__).parent))
    
    core_modules = [
        ("src.core.final_rag_system", "Final RAG System"),
        ("src.core.llm_interface", "LLM Interface"),
        ("src.core.embedding_manager", "Embedding Manager"),
        ("src.core.document_processor", "Document Processor"),
        ("src.core.query_analyzer", "Query Analyzer"),
        ("src.core.conversation_manager", "Conversation Manager"),
        ("src.core.utils", "Utilities"),
    ]
    
    mcp_modules = [
        ("src.mcp.mcp_server", "MCP Server"),
        ("src.mcp.client", "MCP Client"),
        ("src.mcp", "MCP Package"),
    ]
    
    results = []
    all_available = True
    
    results.append("Core Modules:")
    for module_name, display_name in core_modules:
        try:
            importlib.import_module(module_name)
            results.append(f"✅ {display_name}")
        except ImportError as e:
            results.append(f"❌ {display_name}: {e}")
            all_available = False
    
    results.append("\nMCP Modules:")
    for module_name, display_name in mcp_modules:
        try:
            importlib.import_module(module_name)
            results.append(f"✅ {display_name}")
        except ImportError as e:
            results.append(f"❌ {display_name}: {e}")
            all_available = False
    
    return all_available, results

def check_file_structure() -> Tuple[bool, List[str]]:
    """Check required file structure."""
    required_paths = [
        ("src/", "Source directory"),
        ("src/core/", "Core modules"),
        ("src/mcp/", "MCP modules"), 
        ("src/ui/", "UI modules"),
        ("src/examples/", "Examples"),
        ("config/", "Configuration"),
        ("data/", "Data directory"),
        ("docs/", "Documentation"),
    ]
    
    required_files = [
        ("src/core/final_rag_system.py", "Main RAG system"),
        ("src/mcp/mcp_server.py", "MCP server"),
        ("src/mcp/client.py", "MCP client"),
        ("src/ui/streamlit_app_mcp.py", "MCP Streamlit app"),
        ("start_mcp.ps1", "PowerShell launcher"),
        ("launch_mcp_server.py", "Python launcher"),
    ]
    
    results = []
    all_exist = True
    
    results.append("Directory Structure:")
    for path, description in required_paths:
        if os.path.exists(path):
            results.append(f"✅ {path} ({description})")
        else:
            results.append(f"❌ {path} ({description})")
            all_exist = False
    
    results.append("\nRequired Files:")
    for file_path, description in required_files:
        if os.path.exists(file_path):
            results.append(f"✅ {file_path} ({description})")
        else:
            results.append(f"❌ {file_path} ({description})")
            all_exist = False
    
    return all_exist, results

def check_data_directories() -> Tuple[bool, List[str]]:
    """Check and create data directories."""
    data_dirs = [
        "data/documents",
        "data/embeddings",
        "data/logs",
        "data/cache",
        "data/exports"
    ]
    
    results = []
    created_any = False
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            results.append(f"✅ {data_dir}")
        else:
            try:
                os.makedirs(data_dir, exist_ok=True)
                results.append(f"🔧 Created {data_dir}")
                created_any = True
            except Exception as e:
                results.append(f"❌ Failed to create {data_dir}: {e}")
    
    return True, results

def check_ollama_service() -> Tuple[bool, List[str]]:
    """Check configured Ollama service and model availability."""
    results = []

    try:
        with open("config/config.json", "r") as f:
            config = json.load(f)

        uses_ollama = (
            config.get("llm", {}).get("provider") == "ollama"
            or config.get("embedding", {}).get("provider") == "ollama"
        )
        if not uses_ollama:
            return True, ["✅ Ollama not configured; skipping service check"]

        if not importlib.util.find_spec("requests"):
            return False, ["❌ requests package required for Ollama service check"]

        import requests

        base_url = config.get("llm", {}).get("base_url", "http://localhost:11434").rstrip("/")
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
        model_names = [model.get("name", "") for model in response.json().get("models", [])]
        model_bases = {name.split(":")[0] for name in model_names}

        results.append(f"✅ Ollama service reachable at {base_url}")

        checks = [
            ("LLM", config.get("llm", {}).get("model")),
            ("Embedding", config.get("embedding", {}).get("model")),
        ]
        all_found = True
        for label, model in checks:
            if not model:
                continue
            if model in model_names or model in model_bases:
                results.append(f"✅ {label} model configured: {model}")
            else:
                all_found = False
                results.append(f"❌ {label} model not found in Ollama: {model}")

        if model_names:
            results.append(f"Available models: {', '.join(model_names)}")
        else:
            results.append("⚠️  Ollama reported no installed models")

        return all_found, results

    except Exception as e:
        return False, [f"❌ Ollama service check failed: {e}"]

def run_quick_test() -> Tuple[bool, List[str]]:
    """Run a quick functionality test."""
    results = []
    
    try:
        # Test configuration loading
        with open("config/config.json", 'r') as f:
            config = json.load(f)
        
        results.append("✅ Configuration loading")
        
        # Test core module initialization (without actual model loading)
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from core.utils import setup_logging, validate_input
        
        results.append("✅ Utility functions")
        
        # Test MCP module import
        from src.mcp.mcp_server import MCPProtocolHandler
        
        results.append("✅ MCP server import")
        
        # Test basic validation
        if validate_input("test query", config):
            results.append("✅ Input validation")
        else:
            results.append("⚠️  Input validation (strict mode)")
        
        return True, results
        
    except Exception as e:
        results.append(f"❌ Quick test failed: {e}")
        return False, results

def main():
    """Run all health checks."""
    print("🔍 Final RAG Chatbot MCP System Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration", check_configuration),
        ("File Structure", check_file_structure),
        ("Data Directories", check_data_directories),
        ("Ollama Service", check_ollama_service),
        ("Core Modules", check_core_modules),
        ("Quick Test", run_quick_test),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        print("-" * 30)
        
        passed, results = check_func()
        
        if isinstance(results, list):
            for result in results:
                print(f"  {result}")
        else:
            print(f"  {results}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 System Health Check: PASSED")
        print("\n✅ Your Final RAG Chatbot MCP system is ready!")
        print("\n🚀 Quick Start:")
        print("  1. Run: .\\start_mcp.ps1")
        print("  2. Or: python src/examples/mcp_examples.py --interactive")
        print("  3. Or: streamlit run src/ui/streamlit_app_mcp.py")
    else:
        print("⚠️  System Health Check: ISSUES FOUND")
        print("\n🔧 Please fix the issues above before proceeding.")
        print("\n📚 For help, see:")
        print("  - docs/MCP_QUICKSTART.md")
        print("  - docs/README.md")

if __name__ == "__main__":
    main()
