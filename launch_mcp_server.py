#!/usr/bin/env python3
"""
MCP Server Launcher for Final RAG Chatbot

This script provides an easy way to launch the MCP server for the
Final RAG Chatbot system.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Launch Final RAG Chatbot MCP Server")
    parser.add_argument("--config", default="config/config.json", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = Path(__file__).parent
    server_script = script_dir / "src" / "mcp" / "mcp_server.py"
    
    if not server_script.exists():
        print(f"Error: MCP server script not found at {server_script}")
        sys.exit(1)
    
    # Launch the MCP server
    cmd = [
        args.python,
        str(server_script),
        "--config", args.config,
        "--log-level", args.log_level
    ]
    
    print(f"Launching MCP server: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching MCP server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMCP server stopped by user")

if __name__ == "__main__":
    main()
