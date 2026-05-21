"""Focused health tests for the Final RAG Chatbot repository.

These tests avoid real model downloads and live LLM calls by faking the heavy
runtime integrations. They verify the local package layout, configuration,
document parsing, MCP metadata, and the core RAG flow.
"""

import asyncio
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONFIG = ROOT / "config" / "config.json"

sys.path.insert(0, str(SRC))


class FakeArray:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeArray(self.rows[item])
        return self.rows[item]

    def __setitem__(self, item, value):
        self.rows[item] = list(value)

    @property
    def shape(self):
        width = len(self.rows[0]) if self.rows else 0
        return (len(self.rows), width)

    @property
    def nbytes(self):
        return len(self.rows) * (len(self.rows[0]) if self.rows else 0) * 8

    def tolist(self):
        return [row[:] for row in self.rows]


def install_fake_numeric_modules():
    """Install tiny numpy/sklearn stand-ins for dependency-light tests."""

    def as_rows(value):
        if isinstance(value, FakeArray):
            return value.rows
        return value

    def vstack(values):
        rows = []
        for value in values:
            rows.extend(as_rows(value))
        return FakeArray(rows)

    def argsort(values):
        return sorted(range(len(values)), key=lambda idx: values[idx])

    def delete(values, index, axis=0):
        rows = as_rows(values)
        return FakeArray(rows[:index] + rows[index + 1 :])

    def cosine_similarity(left, right):
        left_rows = as_rows(left)
        right_rows = as_rows(right)
        output = []
        for lrow in left_rows:
            scores = []
            for rrow in right_rows:
                dot = sum(a * b for a, b in zip(lrow, rrow))
                lnorm = sum(a * a for a in lrow) ** 0.5
                rnorm = sum(b * b for b in rrow) ** 0.5
                scores.append(dot / (lnorm * rnorm) if lnorm and rnorm else 0.0)
            output.append(scores)
        return output

    numpy_module = types.ModuleType("numpy")
    numpy_module.ndarray = FakeArray
    numpy_module.array = lambda value: FakeArray(value)
    numpy_module.vstack = vstack
    numpy_module.argsort = argsort
    numpy_module.delete = delete

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")
    pairwise_module = types.ModuleType("sklearn.metrics.pairwise")
    pairwise_module.cosine_similarity = cosine_similarity
    metrics_module.pairwise = pairwise_module
    sklearn_module.metrics = metrics_module

    sys.modules.setdefault("numpy", numpy_module)
    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.metrics", metrics_module)
    sys.modules.setdefault("sklearn.metrics.pairwise", pairwise_module)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def fake_ollama_post(url, json=None, timeout=None):
    if url.endswith("/api/chat"):
        return FakeResponse(
            {"message": {"content": "Artificial intelligence is simulated intelligence."}}
        )

    texts = (json or {}).get("input", [])
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        lower = text.lower()
        embeddings.append(
            [
                1.0 if "artificial intelligence" in lower else 0.0,
                1.0 if "machine learning" in lower else 0.0,
                min(len(text), 1000) / 1000.0,
            ]
        )

    return FakeResponse({"embeddings": embeddings})


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG.open("r", encoding="utf-8") as f:
            cls.config = json.load(f)

    def test_config_has_required_sections(self):
        required = {
            "llm",
            "embedding",
            "retrieval",
            "system",
            "mcp",
            "roles",
            "paths",
            "security",
            "performance",
        }
        self.assertTrue(required.issubset(self.config))

    def test_input_validation_rejects_unsafe_content(self):
        from core.utils import validate_input

        self.assertTrue(validate_input("What is artificial intelligence?", self.config))
        self.assertFalse(validate_input("<script>alert('x')</script>", self.config))
        self.assertFalse(validate_input("", self.config))

    def test_document_processor_handles_text_json_and_csv(self):
        from core.document_processor import DocumentProcessor

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "note.txt").write_text(
                "Artificial intelligence helps systems answer questions.",
                encoding="utf-8",
            )
            (temp_path / "data.json").write_text(
                json.dumps({"title": "ML", "content": "Machine learning uses data."}),
                encoding="utf-8",
            )
            (temp_path / "table.csv").write_text(
                "name,description\nRAG,Retrieval augmented generation",
                encoding="utf-8",
            )

            docs = DocumentProcessor(self.config).process_documents(str(temp_path))

        self.assertGreaterEqual(len(docs), 3)
        self.assertTrue(all("content" in doc and "metadata" in doc for doc in docs))
        self.assertEqual({"csv", "json", "text"}, {doc["metadata"]["type"] for doc in docs})

    def test_mcp_initialize_and_tools_list(self):
        install_fake_numeric_modules()
        from mcp.mcp_server import MCPProtocolHandler

        async def run_checks():
            handler = MCPProtocolHandler(str(CONFIG))
            init_response = await handler.handle_request({"id": "1", "method": "initialize"})
            tools_response = await handler.handle_request({"id": "2", "method": "tools/list"})
            return init_response, tools_response

        init_response, tools_response = asyncio.run(run_checks())
        self.assertEqual("Final RAG Chatbot", init_response["result"]["serverInfo"]["name"])
        tool_names = {tool["name"] for tool in tools_response["result"]["tools"]}
        self.assertIn("chat", tool_names)
        self.assertIn("load_documents", tool_names)
        self.assertIn("search_documents", tool_names)
        self.assertIn("clear_documents", tool_names)

    def test_chatbot_loads_documents_and_answers_with_fakes(self):
        install_fake_numeric_modules()
        from core.final_rag_system import FinalRAGChatbot

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "documents"
            docs_dir.mkdir()
            (docs_dir / "ai.txt").write_text(
                "Artificial intelligence is the simulation of human intelligence.",
                encoding="utf-8",
            )

            config = copy.deepcopy(self.config)
            config["paths"] = {
                "data_dir": str(temp_path / "data"),
                "documents_dir": str(docs_dir),
                "embeddings_dir": str(temp_path / "embeddings"),
                "logs_dir": str(temp_path / "logs"),
                "cache_dir": str(temp_path / "cache"),
            }
            config["system"]["cache_embeddings"] = False
            config["system"]["enable_streaming"] = False

            with patch("requests.post", side_effect=fake_ollama_post):
                chatbot = FinalRAGChatbot(config_path=str(CONFIG), custom_config=config)
                self.assertTrue(chatbot.load_documents(str(docs_dir)))
                loaded_once = chatbot.get_stats()["documents_loaded"]
                self.assertTrue(chatbot.load_documents(str(docs_dir)))
                self.assertEqual(loaded_once, chatbot.get_stats()["documents_loaded"])
                response = chatbot.chat("What is artificial intelligence?")

        self.assertIn("simulated intelligence", response)
        self.assertIn("**Sources**", response)

    def test_clear_documents_removes_persisted_cache(self):
        install_fake_numeric_modules()
        from core.final_rag_system import FinalRAGChatbot

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "documents"
            docs_dir.mkdir()
            (docs_dir / "ai.txt").write_text(
                "Artificial intelligence is the simulation of human intelligence.",
                encoding="utf-8",
            )

            config = copy.deepcopy(self.config)
            config["paths"] = {
                "data_dir": str(temp_path / "data"),
                "documents_dir": str(docs_dir),
                "embeddings_dir": str(temp_path / "embeddings"),
                "logs_dir": str(temp_path / "logs"),
                "cache_dir": str(temp_path / "cache"),
            }
            config["system"]["cache_embeddings"] = True
            config["system"]["enable_streaming"] = False

            with patch("requests.post", side_effect=fake_ollama_post):
                chatbot = FinalRAGChatbot(config_path=str(CONFIG), custom_config=config)
                self.assertTrue(chatbot.load_documents(str(docs_dir)))
                cache_path = Path(config["paths"]["embeddings_dir"]) / "embeddings_cache.pkl"
                self.assertTrue(cache_path.exists())

                result = chatbot.clear_documents(delete_cache=True)

        self.assertEqual(0, result["documents_loaded"])
        self.assertTrue(result["cache_deleted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
