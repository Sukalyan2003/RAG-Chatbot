"""
Dependency Checker and Installer

This script checks for required dependencies and offers to install them.
Run this before using the Final RAG Chatbot system.
"""

import subprocess
import sys
import importlib
from pathlib import Path
from typing import List, Tuple

# Required packages with their import names and pip install names
REQUIRED_PACKAGES = [
    ("sklearn", "scikit-learn", "Machine learning utilities"),
    ("bs4", "beautifulsoup4", "Web scraping"),
    ("pdfminer", "pdfminer.six", "PDF processing"),
    ("numpy", "numpy", "Numerical computing"),
    ("pandas", "pandas", "Data manipulation"),
    ("requests", "requests", "HTTP requests"),
    ("streamlit", "streamlit", "Web interface framework"),
    ("plotly", "plotly", "Interactive charts and graphs"),
    ("altair", "altair", "Statistical visualization"),
    ("psutil", "psutil", "System monitoring"),
    ("json", None, "JSON handling (built-in)"),
    ("os", None, "Operating system interface (built-in)"),
    ("pathlib", None, "Path handling (built-in)"),
    ("logging", None, "Logging (built-in)"),
    ("datetime", None, "Date/time handling (built-in)"),
    ("asyncio", None, "Async programming (built-in)"),
    ("pickle", None, "Serialization (built-in)"),
    ("re", None, "Regular expressions (built-in)"),
    ("typing", None, "Type hints (built-in)"),
    ("collections", None, "Collections (built-in)")
]

def check_package(import_name: str) -> bool:
    """Check if a package can be imported."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(pip_name: str) -> bool:
    """Install a package using pip."""
    try:
        print(f"   Installing {pip_name}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Failed to install {pip_name}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print(" Checking Python Version")
    print("-" * 30)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"   Current version: {version_str}")
    
    if version.major >= 3 and version.minor >= 8:
        print("    Python version is compatible")
        return True
    else:
        print("    Python 3.8+ required")
        return False

def check_dependencies() -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Check all dependencies and return missing packages."""
    print("\n Checking Dependencies")
    print("-" * 30)
    
    missing_packages = []
    installed_packages = []
    
    for import_name, pip_name, description in REQUIRED_PACKAGES:
        if pip_name is None:  # Built-in package
            print(f"    {import_name} (built-in)")
            installed_packages.append(import_name)
        else:
            if check_package(import_name):
                print(f"    {import_name}")
                installed_packages.append(import_name)
            else:
                print(f"    {import_name} - {description}")
                missing_packages.append((import_name, pip_name, description))
    
    return missing_packages, installed_packages

def install_missing_packages(missing_packages: List[Tuple[str, str, str]]) -> bool:
    """Install all missing packages."""
    if not missing_packages:
        return True
    
    print(f"\n Installing Missing Packages")
    print("-" * 30)
    
    failed_installations = []
    
    for import_name, pip_name, description in missing_packages:
        if install_package(pip_name):
            print(f"    {import_name} installed successfully")
        else:
            failed_installations.append((import_name, pip_name))
    
    if failed_installations:
        print(f"\n Failed to install:")
        for import_name, pip_name in failed_installations:
            print(f"   - {import_name} ({pip_name})")
        return False
    else:
        print(f"\n All packages installed successfully!")
        return True

def verify_installation():
    """Verify that the Final RAG system can be imported."""
    print(f"\n Verifying System Installation")
    print("-" * 30)
    
    try:
        # Test individual components
        src_path = str(Path(__file__).resolve().parents[1] / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        components = [
            "core.utils",
            "core.document_processor",
            "core.embedding_manager",
            "core.llm_interface",
            "core.query_analyzer",
            "core.conversation_manager",
            "core.final_rag_system",
        ]
        
        for component in components:
            try:
                importlib.import_module(component)
                print(f"    {component}")
            except ImportError as e:
                print(f"    {component}: {e}")
                return False
        
        print(f"\nFinal RAG System verification successful!")
        return True
        
    except Exception as e:
        print(f"    Verification failed: {e}")
        return False

def create_requirements_file():
    """Create a requirements.txt file."""
    print(f"\n Creating requirements.txt")
    print("-" * 30)
    
    try:
        requirements = []
        for import_name, pip_name, description in REQUIRED_PACKAGES:
            if pip_name:  # Not a built-in package
                requirements.append(f"{pip_name}  # {description}")
        
        with open("requirements.txt", "w") as f:
            f.write("# Final RAG Chatbot Dependencies\n")
            f.write("# Install with: pip install -r requirements.txt\n\n")
            f.write("\n".join(requirements))
        
        print(f"    requirements.txt created")
        print(f"    You can install all dependencies with: pip install -r requirements.txt")
        return True
        
    except Exception as e:
        print(f"    Failed to create requirements.txt: {e}")
        return False

def main():
    """Main function."""
    print("Final RAG Chatbot - Dependency Checker")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n Incompatible Python version. Please upgrade to Python 3.8+")
        return False
    
    # Check dependencies
    missing_packages, installed_packages = check_dependencies()
    
    print(f"\n Summary:")
    print(f"    Installed: {len(installed_packages)}")
    print(f"    Missing: {len(missing_packages)}")
    
    if missing_packages:
        print(f"\n Install missing packages? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice in ['y', 'yes']:
            if install_missing_packages(missing_packages):
                print(f"\n All dependencies installed!")
            else:
                print(f"\n Some installations failed. Please install manually.")
                return False
        else:
            print(f"\n️  Some dependencies are missing. The system may not work properly.")
            return False
    
    # Create requirements file
    create_requirements_file()
    
    # Verify installation
    if verify_installation():
        print(f"\n Setup Complete!")
        print(f"    All dependencies are ready")
        print(f"    Final RAG system verified")
        print(f"\n Next Steps:")
        print(f"   1. Run: python tests/test_system.py")
        print(f"   2. Try: python src/examples/basic_usage.py")
        print(f"   3. Read: docs/QUICKSTART.md")
        return True
    else:
        print(f"\n Setup verification failed.")
        print(f"   Please check for errors and try again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
