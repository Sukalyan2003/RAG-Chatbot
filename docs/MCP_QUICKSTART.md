# MCP Quick Start

The Model Context Protocol (MCP) layer exposes the RAG system as a JSON-RPC 2.0 server over stdio. Any MCP-compatible client can drive it.

## What the MCP Layer Adds

- A versioned protocol (`2024-11-05`) with `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, and `prompts/get`.
- Seven tools (`chat`, `load_documents`, `search_documents`, `analyze_query`, `get_stats`, `clear_conversation`, `clear_documents`).
- Resources for conversation history (per role), the live config, and the document index.
- A `system_prompt` prompt template that returns the role-conditioned system prompt the chatbot uses internally.
- An async Python client (`MCPClient`) and a higher-level wrapper (`MCPIntegratedRAG`) for in-process usage.

## Install

```bash
pip install -r requirements.txt
pip install -r mcp-requirements.txt   # only if you want HTTP / WebSocket transports
```

The stdio server itself has no extra dependencies beyond the core requirements.

## Configuration

`config/config.json` includes an `mcp` section:

```json
{
  "mcp": {
    "enabled": true,
    "server_timeout": 30,
    "protocol_version": "2024-11-05",
    "stdio_buffer_size": 8192,
    "http_port": 8080,
    "http_host": "localhost",
    "enable_http_transport": false,
    "enable_websocket_transport": false
  }
}
```

Each role has an `mcp_tools` allowlist (see `config/config.json` and the table in [QUICKSTART.md](QUICKSTART.md)).

## Starting the Server

```bash
# Stdio launcher (cross-platform)
python launch_mcp_server.py --config config/config.json --log-level INFO

# Or invoke the server directly
python src/mcp/mcp_server.py --config config/config.json

# Windows / PowerShell — server + Streamlit
.\start_mcp.ps1
.\start_mcp.ps1 -ServerOnly
.\start_mcp.ps1 -StreamlitOnly
```

Logs go to stderr; stdout is reserved for JSON-RPC frames so a misbehaving log line cannot corrupt the protocol stream.

## Tool Reference

| Tool | Required args | Optional args | Returns |
|------|---------------|---------------|---------|
| `chat` | `query` | `role` (default `User`), `stream` | Markdown response with `**Sources**` footer when attribution is on. |
| `load_documents` | `source` | `role` (default `Admin`), `document_type` (`auto` default) | Human-readable load summary. |
| `search_documents` | `query` | `role`, `max_results` (5), `threshold` (0.3) | JSON list of `{content, source, score, metadata}` (content truncated to 500 chars). |
| `analyze_query` | `query` | `role` | JSON analysis dict from `QueryAnalyzer.analyze_query`. |
| `get_stats` | — | `role` | JSON with counters, uptime, doc/chunk counts, last error. |
| `clear_conversation` | — | `role` | Confirmation string. |
| `clear_documents` | — | `role` (default `Admin`), `delete_cache` (true) | JSON with `documents_loaded`, `chunks_loaded`, `cache_deleted`, `cache_path`. |

## Resource Reference

| URI | Payload |
|-----|---------|
| `rag://conversations/admin` | Admin role history |
| `rag://conversations/expert` | Expert role history |
| `rag://conversations/user` | User role history |
| `rag://conversations/guest` | Guest role history |
| `rag://config` | Current `config/config.json` content |
| `rag://documents/list` | One row per loaded source: `{document, type, chunks, source}` |

## Python Client

### High-level wrapper

```python
import asyncio
from src.mcp.client import MCPIntegratedRAG

async def main():
    async with MCPIntegratedRAG(role="User") as rag:
        print(await rag.chat("What is machine learning?"))
        print(await rag.analyze_query("Explain transformers"))
        print(await rag.search_documents("python", max_results=3))

asyncio.run(main())
```

### Lower-level client

```python
import asyncio, sys
from src.mcp.client import MCPClient

async def main():
    client = MCPClient([sys.executable, "src/mcp/mcp_server.py",
                        "--config", "config/config.json"])
    await client.start_server()
    try:
        print(await client.chat("Hello!"))
        print(await client.get_stats())
        print(await client.get_config())
    finally:
        await client.stop_server()

asyncio.run(main())
```

### Drop-in replacement

```python
from src.mcp.client import FinalRAGChatbotMCP

async with FinalRAGChatbotMCP(role="User") as rag:
    print(await rag.chat("Hello"))
```

## Wiring Into Other MCP Clients

Any client that can spawn a stdio MCP server will work. The command and args you need:

```json
{
  "command": "python",
  "args": [
    "/absolute/path/to/src/mcp/mcp_server.py",
    "--config",
    "/absolute/path/to/config/config.json"
  ]
}
```

If the client uses a different schema (top-level `mcp.servers` vs `mcpServers`, etc.), drop the same command/args under the appropriate key.

## Troubleshooting

- **Server exits immediately** — run `python src/mcp/mcp_server.py --log-level DEBUG` and watch stderr; the most common cause is an unreachable Ollama service.
- **Client hangs** — verify the server prints nothing to stdout except JSON-RPC. The client tolerates accidental log lines but a flood will look like a hang.
- **`Method not found`** — check the spelling against the table above; only those methods are routed.
- **Tool call returns "Error: Query parameter is required"** — confirm the `arguments` object includes the required field; some clients drop empty strings before sending.
- **Bypass the server in tests** — set `use_mcp=False` on `MCPIntegratedRAG`; it falls back to a direct in-process `FinalRAGChatbot`.

## Examples

```bash
python src/examples/mcp_examples.py --example client     # short scripted demo
python src/examples/mcp_examples.py --example chat       # chat-only demo
python src/examples/mcp_examples.py --interactive        # REPL against the server
```
