# Final RAG Chatbot (MCP-Enabled)

A Retrieval-Augmented Generation (RAG) chatbot that runs locally against an Ollama backend, exposes its full surface area through the Model Context Protocol (MCP), and ships with a Streamlit web UI, a CLI, and a small set of runnable examples.

## Highlights

- **Local-first**: Ollama is the default LLM and embedding provider (`http://localhost:11434`). A chat-completion compatible API is supported as an optional fallback.
- **MCP server + client**: JSON-RPC 2.0 over stdio, with tools for chat, document loading, search, query analysis, stats, and cache management.
- **Multi-format ingestion**: PDF (pdfminer.six), DOCX (python-docx, optional), TXT, Markdown, JSON, CSV, and web URLs (requests + BeautifulSoup).
- **Role-aware behavior**: Admin / Expert / User / Guest roles control response length, MCP tool access, and permission gating.
- **Persistent embeddings cache**: pickled cache under `data/embeddings/embeddings_cache.pkl` with a versioned header that guards against embedding-model and vector-dimension mismatches. Duplicate chunks are detected by SHA-256 content hash.
- **Two Streamlit UIs**: a direct-import app (`streamlit_app.py`) and an MCP-backed app (`streamlit_app_mcp.py`).

## Repository Layout

```
.
├── config/config.json              Runtime configuration (LLM, embeddings, retrieval, roles, MCP, security)
├── data/                           Documents, embeddings cache, exports, logs (gitignored)
├── docs/                           User-facing documentation
├── health_check.py                 End-to-end health diagnostic
├── launch_mcp_server.py            Cross-platform MCP server launcher
├── main.py                         Trivial entrypoint
├── mcp-requirements.txt            Optional MCP/HTTP transport dependencies
├── requirements.txt                Core Python dependencies
├── scripts/                        Setup and launch helpers
├── src/
│   ├── core/                       RAG engine (orchestrator, doc processor, embeddings, LLM, query analysis, conversation)
│   ├── mcp/                        MCP protocol handler, stdio server, async client, integration wrapper
│   ├── ui/                         Streamlit apps (direct + MCP)
│   └── examples/                   Runnable examples
├── start_mcp.ps1                   Windows/PowerShell launcher
└── tests/test_system.py            Focused unit/integration tests with fake numeric + Ollama backends
```

## Core Modules

### `src/core/final_rag_system.py`
`FinalRAGChatbot` is the orchestrator. It loads `config/config.json`, instantiates the document processor, embedding manager, LLM interface, query analyzer, and conversation manager, then exposes:

- `load_documents(source, document_type="auto")` — file, directory, or URL; deduplicates against the existing index by `(source, file_name, type, chunk markers, content)`.
- `chat(query, stream=None)` — full RAG pipeline: validate input → permission check → analyze → retrieve → generate → record interaction → update stats.
- `chat_async(query)` — runs `chat` in a thread executor when `performance.enable_async` is on.
- `get_stats()`, `clear_conversation()`, `export_conversation(format)`, `reset_system()`, `clear_documents(delete_cache=True)`, `get_document_summaries()`.
- Source attribution is appended as a Markdown `**Sources**` block; the previous block is stripped before the model sees the next turn.
- WSL-friendly path normalization: `C:\Users\…\file.pdf` pasted into the UI is rewritten to `/mnt/c/Users/…/file.pdf`.

### `src/core/document_processor.py`
`DocumentProcessor` handles file types via a dispatch dict. Text is split on sentence boundaries with configurable chunk size and overlap; long sentences fall back to word-level splitting. JSON inputs prefer well-known content fields (`content`, `text`, `body`, `message`, `description`, `summary`). Web URLs are fetched with a fixed User-Agent, stripped of `<script>`/`<style>`, and chunked the same way.

### `src/core/embedding_manager.py`
`EmbeddingManager` supports two embedding providers:

- `ollama` (default) — POSTs to `/api/embed` (batch) and falls back to `/api/embeddings` (legacy single-prompt) for older Ollama builds.
- `sentence_transformers` — local fallback if you prefer not to run an embedding server.

Retrieval is hybrid by default (`retrieval.hybrid_enabled`). A **dense** leg computes cosine similarity from scikit-learn over a stacked NumPy matrix; a **sparse** leg (`src/core/bm25_index.py`, wrapping `rank_bm25.BM25Okapi`) scores the same chunks lexically. The two rankings — `dense_top_k` and `bm25_top_k` deep — are fused with **reciprocal rank fusion** (`score = Σ 1/(rrf_k + rank)`, see `utils.reciprocal_rank_fusion`), which surfaces exact-term matches (identifiers, code symbols) that dense embeddings miss. The fused candidates oversample a wider pool (`retrieval.max_results * retrieval.rerank_oversample_factor`), feed through `rerank_results`, and trim to `retrieval.max_results`. With `hybrid_enabled` off (or `rank_bm25` not installed) it degrades to the dense-only, threshold-gated path. The BM25 index is kept in sync with `self.documents` on add / update / remove / clear / cache-load. The manager also exposes add / retrieve / update / remove / search-by-metadata and pickle import/export.

### `src/core/llm_interface.py`
`LLMInterface` supports `ollama` (POST `/api/chat`) and `local` / chat-completion compatible providers via the optional `openai` Python package. It streams Ollama responses (NDJSON token deltas) when `generate_response(stream=True)` is requested or `system.enable_streaming` is on; non-Ollama providers fall back to a single-chunk pseudo-stream. The class also exposes helpers used by the broader system: `check_relevance`, `summarize_text`, `extract_keywords`, `classify_query`, `generate_questions`, `evaluate_answer`, and `is_available` health probe.

### `src/core/query_analyzer.py`
Lightweight, regex-driven analysis: intent classification (`question`, `tutorial`, `definition`, `comparison`, `troubleshooting`, `command`, `request`), entity extraction, query type, query enhancement, synonym expansion, simple filter extraction, and safety/length validation.

### `src/core/conversation_manager.py`
Role-keyed history with configurable maximum length and session timeout. Provides `add_interaction`, `get_context`, `get_full_history`, `clear_history`, summaries, topic extraction, similar-query lookup, and export as JSON / TXT / CSV.

### `src/core/utils.py`
Logging (with optional Rich), input validation against a denylist of script/SQL/code-execution patterns, output sanitization (HTML stripping, whitespace normalization, length cap), file hashing, system info, and a `PerformanceMonitor`.

## MCP Layer

### `src/mcp/mcp_server.py`
`MCPProtocolHandler` implements the JSON-RPC 2.0 surface (initialize, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`). It also emits `notifications/progress` for long-running tools (`chat`, `load_documents`) when the caller supplies `_meta.progressToken` on the request — see the "Progress notifications" section below. `MCPStdioServer` reads requests from stdin and writes responses and notifications to stdout (newline-delimited JSON-RPC), with logs routed to stderr so they don't corrupt the protocol stream.

### Progress notifications

`chat` and `load_documents` accept an optional `_meta.progressToken` field. When supplied, the server emits notifications shaped as:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "prog-abcd1234",
    "progress": 64,
    "total": 512,
    "stage": "embedding",
    "message": "embedding: 64/512"
  }
}
```

Stages for `load_documents`: `reading`, `chunking`, `dedup`, `embedding`, `storing`, `saving_cache`. Stages for `chat`: `validating`, `analyzing`, `retrieving`, `context`, `generating`, `done`. The bundled `MCPClient` exposes this through a `progress_callback=(stage, current, total) -> None` argument; the persistent reader task demultiplexes responses from notifications so callers see real-time progress without managing tokens directly.

Tools (7):

| Tool | Purpose |
|------|---------|
| `chat` | Run a full RAG turn for the given role. |
| `load_documents` | Ingest a file, directory, or URL. |
| `search_documents` | Top-K semantic retrieval with a similarity threshold. |
| `analyze_query` | Return the analyzed/enhanced query. |
| `get_stats` | Performance counters, uptime, document/chunk counts, last error. |
| `clear_conversation` | Drop role-scoped chat history. |
| `clear_documents` | Drop loaded documents/embeddings and (optionally) delete the persisted cache file. |

Resources:

- `rag://conversations/{admin|expert|user|guest}` — role-scoped history.
- `rag://config` — current `config/config.json`.
- `rag://documents/list` — summarized document index (one row per source).

Prompts:

- `system_prompt` — returns the role-conditioned system prompt the chatbot uses internally.

### `src/mcp/client.py`
- `MCPClient` — async subprocess client that speaks stdio JSON-RPC, tolerates accidental non-JSON lines on stdout, and exposes typed helpers for each tool/resource/prompt.
- `MCPIntegratedRAG` — same surface as `FinalRAGChatbot` but async; usable as an `async with` context manager.
- `FinalRAGChatbotMCP` — drop-in subclass of `MCPIntegratedRAG` matching the original constructor signature.

## Configuration

`config/config.json` drives everything. Key sections:

- `llm` — provider (`ollama` default), `base_url`, `model` (`qwen3:4b-instruct` by default), temperature, `max_tokens`, `num_ctx` (`"auto"` by default — auto-tuner picks 2048/4096/8192 based on detected VRAM), timeout. `context_window` is accepted as a deprecated alias for `num_ctx`.
- `embedding` — provider (`ollama` default), `model` (`qwen3-embedding:0.6b` by default), batch size (POSTs one HTTP request per batch to `/api/embed`), max length, device, timeout.
- `retrieval` — `similarity_threshold` (0.5), `max_results` (3 — the reranker means smaller K still wins), `chunk_size` (1000), `chunk_overlap` (200), `rerank_results`, `rerank_oversample_factor` (4), `hybrid_enabled` (true), `bm25_top_k` (20), `dense_top_k` (20), `rrf_k` (60).
- `system` — log level, conversation history cap, source attribution, streaming, cache flag, session timeout, `auto_tune` (default `true`), and `ollama_env` block (`keep_alive`, `kv_cache_type`, `num_gpu` — each accepts `"auto"` to opt in to hardware-derived defaults). With `auto_tune=true`, the engine probes `nvidia-smi` for VRAM and picks tight (≤5 GB → q8_0 KV cache, num_ctx=2048), mid (≤9 GB → f16, num_ctx=4096), ample (>9 GB → f16, num_ctx=8192), or cpu (no GPU → num_gpu=0). Explicit config values always win.
- `mcp` — `enabled`, server timeout, protocol version (`2024-11-05`), stdio buffer size, optional HTTP/WebSocket transport flags.
- `roles` — per-role permissions, response length, access level, allowed MCP tools.
- `paths` — `data_dir`, `documents_dir`, `embeddings_dir`, `logs_dir`, `cache_dir`.
- `security` — input validation toggle, max query length, rate limit, audit logging flag.
- `performance` — async toggle, worker threads, batch processing, memory cap.

## Install & Run

```bash
python3 -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r mcp-requirements.txt  # optional: HTTP/WebSocket MCP transport

python health_check.py               # diagnose deps, config, Ollama service, modules
```

Make sure Ollama is running locally and the configured models are pulled:

```bash
ollama serve
ollama pull qwen3:4b-instruct
ollama pull qwen3-embedding:0.6b
```

Then choose one of:

```bash
# MCP-backed Streamlit UI
streamlit run src/ui/streamlit_app_mcp.py

# Direct-import Streamlit UI
streamlit run src/ui/streamlit_app.py

# Interactive CLI
python -m src.core.final_rag_system --interactive --role User

# MCP server alone (stdio)
python launch_mcp_server.py --config config/config.json
```

On Windows, `start_mcp.ps1` wraps the server + Streamlit combo (`-ServerOnly` / `-StreamlitOnly` switches available).

## Programmatic Usage

```python
from src.core.final_rag_system import FinalRAGChatbot

with FinalRAGChatbot(role="User") as bot:
    bot.load_documents("data/documents/")
    print(bot.chat("What is machine learning?"))
    print(bot.get_stats())
```

```python
import asyncio
from src.mcp.client import MCPIntegratedRAG

async def main():
    async with MCPIntegratedRAG(role="User") as rag:
        print(await rag.chat("Summarize the loaded docs"))
        print(await rag.get_stats())

asyncio.run(main())
```

## Testing

```bash
python -m unittest tests.test_system
```

The test suite installs lightweight stand-ins for NumPy / scikit-learn and fakes the Ollama HTTP surface, so it runs without a live model. It covers config shape, input validation, the document processor across text/JSON/CSV, the MCP initialize + `tools/list` handshake, end-to-end load → chat with deduplication, and the `clear_documents` cache cleanup path.

## Further Reading

- [QUICKSTART.md](QUICKSTART.md) — fastest path to a working chatbot.
- [MCP_QUICKSTART.md](MCP_QUICKSTART.md) — MCP-specific setup and integration.
- [MCP_CONVERSION_SUMMARY.md](MCP_CONVERSION_SUMMARY.md) — what the MCP layer adds and how to wire it into any MCP-compatible client.
- [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) — the web UIs in detail.
