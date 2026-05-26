# Streamlit UI Guide

The project ships two Streamlit interfaces. Pick the one that matches how you want to use the system.

## Which UI Should I Run?

| File | Backend | Use when |
|------|---------|----------|
| `src/ui/streamlit_app_mcp.py` | MCP server in a subprocess | You want the exact same surface as any other MCP client, async I/O, and isolation from the UI process. |
| `src/ui/streamlit_app.py` | Direct in-process imports | You want a lightweight UI with no subprocess, and you do not need the MCP transport. |

Both share the same configuration, document directory, and embeddings cache. They differ only in how they talk to the RAG engine.

## Launching

```bash
streamlit run src/ui/streamlit_app_mcp.py
streamlit run src/ui/streamlit_app.py

# Custom port / bind
streamlit run src/ui/streamlit_app_mcp.py --server.port 8502 --server.address 0.0.0.0
```

On Windows, `start_mcp.ps1` runs the MCP UI by default (`-StreamlitOnly` skips the background MCP server, which is fine because the UI starts its own).

## Layout

The MCP UI has three tabs and a sidebar:

- **💬 Chat** — system metrics row (queries, average response time, success rate, errors), the conversation transcript, and the chat input. Assistant messages include an expandable "Response Details" panel with query analysis and search counts.
- **📊 Analytics** — refreshable stats table and a successful-vs-error bar chart.
- **📄 Documents** — refreshable list of loaded documents grouped by source: `Document`, `Type`, `Chunks`, `Source`.

Sidebar controls:

- **Role selector** — `User` / `Expert` / `Admin`. Switches the active role for subsequent chat turns.
- **Use MCP Protocol** — toggle the MCP backend on/off. Off uses the in-process `FinalRAGChatbot`.
- **Upload Document** — accepts PDF, TXT, MD, JSON, CSV (and DOCX if `python-docx` is installed). Saved under `data/documents/uploads/` with a sanitized filename, then loaded.
- **Document Directory** — paste a path (Windows paths pasted into WSL get rewritten automatically). Loads every supported file under it.
- **Clear Conversation** — drops history for the active role.
- **Clear Documents / Cache** — drops loaded documents/embeddings and deletes `data/embeddings/embeddings_cache.pkl`.
- **Refresh Stats** — re-pulls counters from the engine.
- **Export Chat** — turns the current session into a CSV download.

## Streaming responses

Set `system.enable_streaming: true` in `config/config.json` to render Ollama answers token-by-token.

- **Direct UI (`streamlit_app.py`)** — tick the "🔄 Streaming" checkbox on the chat page. `FinalRAGChatbot.chat(stream=True)` returns an iterator of text deltas which the UI hands to `st.write_stream`. The `**Sources**` block is appended as the final delta after the model finishes.
- **MCP UI (`streamlit_app_mcp.py`)** — when `system.enable_streaming` is on **and** the "Use MCP Protocol" toggle is off, the UI streams from the in-process engine via `MCPIntegratedRAG.chat_stream`. With MCP on, streaming over JSON-RPC notifications is deferred (Phase 13 in the roadmap), so the UI renders the full answer once the MCP `chat` tool returns.

## Session State

The MCP UI keeps one asyncio event loop per Streamlit session in `st.session_state.async_loop`. This is important: MCP child-process pipes must be polled from the same loop that created them. Don't replace it with `asyncio.run(...)` calls — each new loop would close the previous pipes.

## Direct UI (`streamlit_app.py`)

The direct UI is a single class (`StreamlitRAGApp`) that runs an initial smoke test, then exposes the same conceptual pages — dashboard, chat, document management, role switcher, configuration editor, analytics, tests, and exports — directly against `FinalRAGChatbot`. It is heavier on Plotly visualizations and lighter on async plumbing.

## Configuration Edits From the UI

The direct UI surfaces editable LLM, embedding, retrieval, system, and security settings backed by `config/config.json`. Changes are applied to the in-memory copy used by the running chatbot; persisting back to disk requires explicit save. Prefer editing `config/config.json` directly when you want changes to outlive the session.

## Troubleshooting

- **UI launches but chat hangs** — Ollama is unreachable or the configured model is not pulled. `python health_check.py` confirms.
- **Subprocess never returns (MCP UI)** — the MCP server is logging to stdout instead of stderr. Make sure you launched it via `mcp_server.py`'s `main()`, which configures logging correctly.
- **`No module named 'mcp'` inside Streamlit** — the local `src/mcp/` package is being shadowed by a third-party package. The MCP UI prepends `src/` to `sys.path` to win the lookup; if you run a custom entrypoint, do the same.
- **Stale documents listed** — click "Refresh Document List" on the Documents tab; the list is cached in session state.

## Useful Commands

```bash
# Run the UI with verbose logs
streamlit run src/ui/streamlit_app_mcp.py --logger.level debug

# Health check before launching
python health_check.py

# Run the test suite without launching the UI
python -m unittest tests.test_system
```
