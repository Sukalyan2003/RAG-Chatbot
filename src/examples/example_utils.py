"""Shared helpers for runnable examples."""

import json
from pathlib import Path
from typing import Any


SAMPLE_DOCS = {
    "sample_ai.txt": """
# Artificial Intelligence Overview

Artificial intelligence is the simulation of human intelligence in machines.
Key areas include machine learning, natural language processing, computer vision,
robotics, and expert systems.
""",
    "sample_python.txt": """
# Python Programming

Python is a high-level, interpreted programming language known for readable
syntax, dynamic typing, a large standard library, cross-platform support, and a
large ecosystem for web development, automation, data science, and AI.
""",
    "sample_ml.txt": """
# Machine Learning Fundamentals

Machine learning systems learn patterns from data. Common types include
supervised learning with labeled data, unsupervised learning for pattern
discovery, and reinforcement learning through feedback.
""",
}


def ensure_sample_documents(documents_dir: str = "data/documents") -> Path:
    """Create small sample documents used by examples if they are missing."""
    docs_path = Path(documents_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        path = docs_path / filename
        if not path.exists():
            path.write_text(content.strip(), encoding="utf-8")

    return docs_path


def response_to_text(response: Any) -> str:
    """Normalize string or streaming-style dict responses for display."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return json.dumps(response, indent=2, default=str)
