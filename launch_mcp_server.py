#!/usr/bin/env python3
"""
MCP Server Launcher for Final RAG Chatbot

This script provides an easy way to launch the MCP server for the
Final RAG Chatbot system.
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path


def _apply_ollama_env_from_config(config_path: Path) -> None:
    """Auto-tune + export ``system.ollama_env`` so the subprocess inherits.

    Reuses the same ``resolve_ollama_tuning`` / ``apply_ollama_env`` pair
    that ``FinalRAGChatbot`` uses, so the launcher and the engine agree on
    the chosen values. Note: env vars matter to ``ollama serve``; users
    who want them honored by the Ollama server should export the same
    vars before starting it.
    """
    try:
        with config_path.open("r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    # Make the engine's utils importable without installing the package.
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    try:
        from core.utils import resolve_ollama_tuning, apply_ollama_env
    except ImportError:
        return

    tuning = resolve_ollama_tuning(config)
    apply_ollama_env(tuning["ollama_env"])
    if tuning["auto_tune"]:
        print(
            f"Auto-tune ({tuning['tier']}): num_ctx={tuning['num_ctx']}, "
            f"ollama_env={tuning['ollama_env']}",
            file=sys.stderr,
        )


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

    _apply_ollama_env_from_config(Path(args.config))

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
