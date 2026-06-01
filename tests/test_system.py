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

    def test_cache_mismatch_rejected_on_load(self):
        install_fake_numeric_modules()
        from core.embedding_manager import EmbeddingManager

        config = copy.deepcopy(self.config)
        config["embedding"]["provider"] = "ollama"
        config["embedding"]["model"] = "configured-model"

        manager = EmbeddingManager(config)
        manager.documents = [{"content": "x", "metadata": {}}]
        cached_data = {
            "schema_version": 1,
            "model_name": "different-model",
            "dim": 3,
            "documents": [{"content": "x", "metadata": {}}],
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        loaded = manager.load_cached_embeddings(cached_data)

        self.assertFalse(loaded)
        self.assertEqual(0, len(manager.documents))
        self.assertIsNone(manager.embeddings)

    def test_cache_match_loads_successfully(self):
        install_fake_numeric_modules()
        from core.embedding_manager import EmbeddingManager

        config = copy.deepcopy(self.config)
        config["embedding"]["provider"] = "ollama"
        config["embedding"]["model"] = "configured-model"

        manager = EmbeddingManager(config)
        cached_data = {
            "schema_version": 1,
            "model_name": "configured-model",
            "dim": 3,
            "documents": [{"content": "x", "metadata": {}}],
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        loaded = manager.load_cached_embeddings(cached_data)

        self.assertTrue(loaded)
        self.assertEqual(1, len(manager.documents))

    def test_content_hash_dedup_skips_identical_chunk(self):
        install_fake_numeric_modules()
        from core.embedding_manager import EmbeddingManager

        config = copy.deepcopy(self.config)
        config["embedding"]["provider"] = "ollama"

        manager = EmbeddingManager(config)
        doc = {"content": "Repeated chunk content", "metadata": {"source": "a.txt"}}

        with patch("requests.post", side_effect=fake_ollama_post):
            self.assertTrue(manager.add_documents([doc.copy()]))
            self.assertEqual(1, manager.last_added_count)
            self.assertEqual(0, manager.last_skipped_count)

            self.assertTrue(manager.add_documents([doc.copy()]))
            self.assertEqual(0, manager.last_added_count)
            self.assertEqual(1, manager.last_skipped_count)

        self.assertEqual(1, len(manager.documents))
        self.assertEqual(
            16, len(manager.documents[0]["metadata"]["content_hash"])
        )

    def test_retrieve_documents_uses_reranker_when_enabled(self):
        install_fake_numeric_modules()
        from core.embedding_manager import EmbeddingManager

        config = copy.deepcopy(self.config)
        config["embedding"]["provider"] = "ollama"
        config["retrieval"]["rerank_results"] = True
        config["retrieval"]["rerank_oversample_factor"] = 4
        config["retrieval"]["similarity_threshold"] = 0.0

        manager = EmbeddingManager(config)
        docs = [
            {
                "content": f"chunk about artificial intelligence #{i}",
                "metadata": {"source": f"doc_{i}.txt", "type": "text"},
            }
            for i in range(6)
        ]

        with patch("requests.post", side_effect=fake_ollama_post):
            self.assertTrue(manager.add_documents(docs))
            calls = {"count": 0}

            real_rerank = manager.rerank_results

            def spying_rerank(results, query):
                calls["count"] += 1
                calls["candidate_count"] = len(results)
                return real_rerank(results, query)

            manager.rerank_results = spying_rerank

            results = manager.retrieve_documents(
                "artificial intelligence", max_results=2, threshold=0.0
            )

        self.assertEqual(1, calls["count"])
        self.assertGreater(calls["candidate_count"], 2)
        self.assertLessEqual(len(results), 2)

    def test_mcp_load_documents_emits_progress_notifications(self):
        install_fake_numeric_modules()
        from mcp.mcp_server import MCPProtocolHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_dir = temp_path / "documents"
            docs_dir.mkdir()
            (docs_dir / "ai.txt").write_text(
                "Artificial intelligence is the simulation of human intelligence.",
                encoding="utf-8",
            )

            override = copy.deepcopy(self.config)
            override["paths"] = {
                "data_dir": str(temp_path / "data"),
                "documents_dir": str(docs_dir),
                "embeddings_dir": str(temp_path / "embeddings"),
                "logs_dir": str(temp_path / "logs"),
                "cache_dir": str(temp_path / "cache"),
            }
            override["system"]["cache_embeddings"] = False
            override["system"]["enable_streaming"] = False

            notifications = []

            def notify(method, params):
                notifications.append((method, params))

            async def run():
                handler = MCPProtocolHandler(str(CONFIG))
                # Reuse our temp config by pre-instantiating the rag_system.
                from core.final_rag_system import FinalRAGChatbot

                with patch("requests.post", side_effect=fake_ollama_post):
                    handler.rag_system = FinalRAGChatbot(
                        role="Admin",
                        config_path=str(CONFIG),
                        custom_config=override,
                    )
                    return await handler.handle_request(
                        {
                            "id": "42",
                            "method": "tools/call",
                            "params": {
                                "name": "load_documents",
                                "arguments": {
                                    "source": str(docs_dir),
                                    "role": "Admin",
                                },
                                "_meta": {"progressToken": "tok-1"},
                            },
                        },
                        notify=notify,
                    )

            with patch("requests.post", side_effect=fake_ollama_post):
                response = asyncio.run(run())

        self.assertNotIn("error", response)
        # Server should have emitted at least one notifications/progress event for our token.
        progress_events = [
            params
            for method, params in notifications
            if method == "notifications/progress" and params.get("progressToken") == "tok-1"
        ]
        self.assertGreater(len(progress_events), 0)
        stages = {event.get("stage") for event in progress_events}
        # Expect at least the chunking and embedding stages.
        self.assertIn("embedding", stages)

    def test_llm_interface_streams_ndjson_chunks(self):
        from core.llm_interface import LLMInterface

        ndjson_lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": ", "}, "done": False}),
            json.dumps({"message": {"content": "world"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "eval_count": 3,
                    "prompt_eval_count": 5,
                }
            ),
        ]

        class FakeStreamResponse:
            def __init__(self, lines):
                self._lines = [line.encode("utf-8") for line in lines]

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield from self._lines

            def close(self):
                return None

        def fake_post(url, json=None, timeout=None, stream=False):
            assert stream is True
            assert (json or {}).get("stream") is True
            return FakeStreamResponse(ndjson_lines)

        config = copy.deepcopy(self.config)
        config["llm"]["provider"] = "ollama"

        llm = LLMInterface(config)
        with patch("requests.post", side_effect=fake_post):
            chunks = list(
                llm.generate_response(
                    query="hi", context="ctx", stream=True
                )
            )

        deltas = [c["content"] for c in chunks if not c["done"]]
        self.assertEqual(["Hello", ", ", "world"], deltas)
        final = chunks[-1]
        self.assertTrue(final["done"])
        self.assertEqual(3, final["stats"].get("eval_count"))
        self.assertEqual(5, final["stats"].get("prompt_eval_count"))

    def test_chat_stream_yields_tokens_and_appends_sources(self):
        install_fake_numeric_modules()
        from core.final_rag_system import FinalRAGChatbot

        ndjson_lines = [
            json.dumps({"message": {"content": "Artificial "}, "done": False}),
            json.dumps({"message": {"content": "intelligence."}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        class FakeStreamResponse:
            def __init__(self, lines):
                self._lines = [line.encode("utf-8") for line in lines]

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield from self._lines

            def close(self):
                return None

        def fake_post(url, json=None, timeout=None, stream=False):
            if url.endswith("/api/chat") and stream:
                return FakeStreamResponse(ndjson_lines)
            # Reuse the canned non-stream Ollama helper for embeds and the
            # non-streaming chat fallback used during ingest.
            return fake_ollama_post(url, json=json, timeout=timeout)

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
            config["system"]["enable_streaming"] = True

            with patch("requests.post", side_effect=fake_post):
                chatbot = FinalRAGChatbot(config_path=str(CONFIG), custom_config=config)
                self.assertTrue(chatbot.load_documents(str(docs_dir)))
                stream = chatbot.chat(
                    "What is artificial intelligence?", stream=True
                )
                deltas = list(stream)

        text = "".join(deltas)
        # Token deltas plus the appended source block must show up in order.
        self.assertIn("Artificial intelligence.", text)
        self.assertIn("**Sources**", text)
        # The source block must arrive after the model deltas, not be
        # interleaved with them.
        self.assertLess(text.index("Artificial intelligence."), text.index("**Sources**"))

    def test_ollama_embed_one_post_per_batch(self):
        install_fake_numeric_modules()
        from core.embedding_manager import EmbeddingManager

        config = copy.deepcopy(self.config)
        config["embedding"]["provider"] = "ollama"
        config["embedding"]["batch_size"] = 32

        # 64 chunks → expect exactly 2 POSTs to /api/embed (one per batch).
        docs = [
            {"content": f"chunk {i} discussing artificial intelligence",
             "metadata": {"source": f"doc_{i}.txt"}}
            for i in range(64)
        ]

        embed_call_count = {"n": 0, "batch_sizes": []}

        def counting_post(url, json=None, timeout=None, **kwargs):
            if url.endswith("/api/embed"):
                embed_call_count["n"] += 1
                embed_call_count["batch_sizes"].append(len((json or {}).get("input", [])))
            return fake_ollama_post(url, json=json, timeout=timeout)

        manager = EmbeddingManager(config)
        with patch("requests.post", side_effect=counting_post):
            self.assertTrue(manager.add_documents(docs))

        self.assertEqual(2, embed_call_count["n"])
        self.assertEqual([32, 32], embed_call_count["batch_sizes"])

    def test_num_ctx_flows_into_ollama_options(self):
        from core.llm_interface import LLMInterface

        config = copy.deepcopy(self.config)
        config["llm"]["provider"] = "ollama"
        config["llm"]["num_ctx"] = 1234
        # Confirm legacy context_window is ignored when num_ctx is set.
        config["llm"]["context_window"] = 8192

        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["payload"] = json
            return FakeResponse({"message": {"content": "ok"}})

        llm = LLMInterface(config)
        with patch("requests.post", side_effect=fake_post):
            llm.generate_response(query="hi")

        self.assertEqual(1234, captured["payload"]["options"]["num_ctx"])

    def test_num_ctx_falls_back_to_context_window(self):
        from core.llm_interface import LLMInterface

        config = copy.deepcopy(self.config)
        config["llm"]["provider"] = "ollama"
        config["llm"].pop("num_ctx", None)
        config["llm"]["context_window"] = 4321

        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["payload"] = json
            return FakeResponse({"message": {"content": "ok"}})

        llm = LLMInterface(config)
        with patch("requests.post", side_effect=fake_post):
            llm.generate_response(query="hi")

        self.assertEqual(4321, captured["payload"]["options"]["num_ctx"])

    def test_resolve_ollama_tuning_picks_tier_defaults(self):
        from core.utils import resolve_ollama_tuning

        # All values "auto" → fully derived from the (mocked) probe.
        config = {
            "system": {
                "auto_tune": True,
                "ollama_env": {
                    "keep_alive": "auto",
                    "kv_cache_type": "auto",
                    "num_gpu": "auto",
                },
            },
            "llm": {"num_ctx": "auto"},
        }

        # 4 GB-class card → tight tier defaults.
        with patch("core.utils.detect_hardware", return_value={
            "gpu_vram_gb": 4.0, "ram_gb": 16.0, "cpu_count": 8,
        }):
            tuning = resolve_ollama_tuning(config)
        self.assertEqual("tight", tuning["tier"])
        self.assertEqual(2048, tuning["num_ctx"])
        self.assertEqual(
            {"keep_alive": "24h", "kv_cache_type": "q8_0", "num_gpu": 999},
            tuning["ollama_env"],
        )

        # No GPU → cpu tier defaults.
        with patch("core.utils.detect_hardware", return_value={
            "gpu_vram_gb": 0.0, "ram_gb": 16.0, "cpu_count": 8,
        }):
            tuning = resolve_ollama_tuning(config)
        self.assertEqual("cpu", tuning["tier"])
        self.assertEqual(0, tuning["ollama_env"]["num_gpu"])

    def test_resolve_ollama_tuning_explicit_overrides_auto(self):
        from core.utils import resolve_ollama_tuning

        config = {
            "system": {
                "auto_tune": True,
                "ollama_env": {
                    "keep_alive": "auto",
                    "kv_cache_type": "q4_0",   # explicit — must not be overridden
                    "num_gpu": "auto",
                },
            },
            "llm": {"num_ctx": 1024},           # explicit — must not be overridden
        }

        with patch("core.utils.detect_hardware", return_value={
            "gpu_vram_gb": 4.0, "ram_gb": 16.0, "cpu_count": 8,
        }):
            tuning = resolve_ollama_tuning(config)

        self.assertEqual(1024, tuning["num_ctx"])
        self.assertEqual("q4_0", tuning["ollama_env"]["kv_cache_type"])
        # The other two come from the tight-tier defaults.
        self.assertEqual("24h", tuning["ollama_env"]["keep_alive"])
        self.assertEqual(999, tuning["ollama_env"]["num_gpu"])

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
