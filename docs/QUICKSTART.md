# Quick Start

Fastest path from a clean checkout to a working chatbot.

## 1. Prerequisites

- Python 3.8+
- A local Ollama install (default backend). [Install instructions](https://ollama.com/download).
- About 4 GB of free RAM. More if you pull larger models.

## 2. Install Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r mcp-requirements.txt  # optional, for HTTP/WebSocket MCP transport
```

## 3. Start Ollama and Pull the Models

```bash
ollama serve
ollama pull qwen3:4b-instruct          # default chat model
ollama pull qwen3-embedding:0.6b       # default embedding model
```

You can change the models in `config/config.json` (`llm.model`, `embedding.model`). Changing `embedding.model` automatically invalidates the persisted cache on next start; the cache header stores the model name and vector dimension and refuses to load a mismatched cache.

## 4. Verify the Install

```bash
python health_check.py
```

This walks through Python version, required packages, optional packages, config validation, file structure, data directories, Ollama reachability, model availability, core/MCP module imports, and a small smoke test.

## 5. Run It

Pick one entrypoint:

```bash
# Streamlit UI backed by the MCP server (recommended)
streamlit run src/ui/streamlit_app_mcp.py

# Streamlit UI with direct in-process imports
streamlit run src/ui/streamlit_app.py

# Interactive CLI
python -m src.core.final_rag_system --interactive --role User

# Run the MCP server alone (stdio)
python launch_mcp_server.py --config config/config.json
```

Streamlit serves at `http://localhost:8501` by default.

## 6. Basic Programmatic Usage

```python
from src.core.final_rag_system import FinalRAGChatbot

with FinalRAGChatbot(role="User") as bot:
    bot.load_documents("data/documents/")   # file, directory, or URL
    print(bot.chat("What is artificial intelligence?"))
    print(bot.get_stats())
```

`load_documents` accepts:

- A single file path (`.pdf`, `.docx`, `.txt`, `.md`, `.json`, `.csv`).
- A directory (walked recursively, dotfiles skipped).
- A `http://` or `https://` URL.
- A list of any of the above.

Duplicate chunks are detected on reload, so you can re-run loading safely.

## 7. Roles

| Role | Response length | MCP tools |
|------|-----------------|-----------|
| `Admin` | detailed | chat, load_documents, get_stats, clear_conversation, clear_documents, analyze_query, search_documents |
| `Expert` | detailed | chat, get_stats, analyze_query, search_documents |
| `User` | concise | chat, get_stats, analyze_query, search_documents |
| `Guest` | brief | chat, search_documents |

Switch role per call: `FinalRAGChatbot(role="Admin")` or `await rag.chat(query)` after setting `rag.role = "Admin"`.

## 8. Async / MCP Path

```python
import asyncio
from src.mcp.client import MCPIntegratedRAG

async def main():
    async with MCPIntegratedRAG(role="User") as rag:
        print(await rag.chat("Summarize the loaded documents"))
        stats = await rag.get_stats()
        print(stats)

asyncio.run(main())
```

## 9. Examples

```bash
python src/examples/basic_usage.py            # text/PDF ingestion + chat
python src/examples/document_processing.py    # multi-format walkthrough
python src/examples/advanced_features.py      # role gating, analytics, exports
python src/examples/mcp_examples.py --interactive   # MCP demo loop
```

## 10. Troubleshooting

- **`Ollama service check failed`** — start `ollama serve`, confirm the URL in `config/config.json`, and pull the configured models.
- **`requests package required`** — the active venv is missing core deps; rerun `pip install -r requirements.txt`.
- **`Sentence transformers not available`** — only required if you switched `embedding.provider` to `sentence_transformers`; `pip install sentence-transformers` or switch back to `ollama`.
- **Stale embeddings** — delete `data/embeddings/embeddings_cache.pkl` or call `chatbot.clear_documents(delete_cache=True)`. A model-mismatch cache is auto-rejected on load with a `WARNING` log, but you still need to re-ingest the documents.
- **Logs** — daily files under `data/logs/rag_system_YYYYMMDD.log`. Bump `system.log_level` to `DEBUG` for verbose tracing.

## 11. Configuration Cheatsheet

```python
from src.core.final_rag_system import FinalRAGChatbot

bot = FinalRAGChatbot(custom_config={
    "system": {"log_level": "DEBUG"},
    "embedding": {"provider": "ollama", "model": "qwen3-embedding:0.6b"},
    "retrieval": {"max_results": 8, "similarity_threshold": 0.25},
})
```

`custom_config` is shallow-merged into the JSON config at startup. To change anything reliably, edit `config/config.json` directly.

---

Next stops: [MCP_QUICKSTART.md](MCP_QUICKSTART.md) for the MCP path, [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for the UI, [INTERVIEW.md](INTERVIEW.md) for deep Q&A.
