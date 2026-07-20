"""
Launcher Script for Final RAG Chatbot System

This script provides an easy way to launch the Streamlit web interface
with proper dependency checking and system initialization.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit_installed():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not present."""
    print(" Installing Streamlit...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "altair"], check=True)
        print(" Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print(" Failed to install Streamlit")
        return False

def run_dependency_check():
    """Run the dependency checker if available."""
    repo_root = Path(__file__).resolve().parents[1]
    dependency_script = repo_root / "scripts" / "setup_dependencies.py"
    if dependency_script.exists():
        print(" Running dependency check...")
        try:
            result = subprocess.run([sys.executable, str(dependency_script)],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(" Dependencies check completed!")
            else:
                print("️ Some dependencies may be missing")
                print(result.stdout)
        except subprocess.TimeoutExpired:
            print("⏰ Dependency check timed out")
        except Exception as e:
            print(f" Error running dependency check: {e}")

def launch_streamlit():
    """Launch the Streamlit application."""
    print(" Launching Final RAG Chatbot System...")
    print(" The web interface will open at: http://localhost:8501")
    print(" Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        repo_root = Path(__file__).resolve().parents[1]
        app_path = repo_root / "src" / "ui" / "streamlit_app.py"
        os.chdir(repo_root)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n Application stopped by user")
    except FileNotFoundError:
        print(" src/ui/streamlit_app.py not found")
    except Exception as e:
        print(f" Error launching Streamlit: {e}")

def main():
    """Main launcher function."""
    print("Final RAG Chatbot System Launcher")
    print("=" * 40)
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "src" / "ui" / "streamlit_app.py"
    
    # Check if we're in the right directory
    if not app_path.exists():
        print(" src/ui/streamlit_app.py not found!")
        print("Please run this script from this repository checkout")
        return
    
    # Check Streamlit installation
    if not check_streamlit_installed():
        print(" Streamlit not found. Installing...")
        if not install_streamlit():
            print(" Cannot proceed without Streamlit")
            return
    
    # Run dependency check
    run_dependency_check()
    
    # Launch the application
    launch_streamlit()

if __name__ == "__main__":
    main()
