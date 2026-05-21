# MCP Layer Summary

The RAG engine and the MCP transport are decoupled. Everything in `src/core/` can be used directly; everything in `src/mcp/` is an additive transport layer on top.

## What `src/mcp/` Provides

### Server

- `MCPProtocolHandler` — request router for JSON-RPC 2.0 methods (`initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`). Lazily constructs a single `FinalRAGChatbot` and switches its `role` per request rather than spinning up a chatbot per call.
- `MCPStdioServer` — reads newline-delimited JSON-RPC requests from stdin, writes responses to stdout, routes logs to stderr.
- `main()` in `mcp_server.py` — CLI entrypoint with `--config` and `--log-level`.

### Client

- `MCPClient` — async subprocess client. Handles request IDs, ignores accidental non-JSON stdout lines, terminates the child process cleanly, exposes typed helpers for every tool/resource/prompt.
- `MCPIntegratedRAG` — async context manager with the same surface as `FinalRAGChatbot`. Has a `use_mcp=False` mode that falls through to a direct `FinalRAGChatbot` instance for tests or environments where a subprocess is undesirable.
- `FinalRAGChatbotMCP` — subclass of `MCPIntegratedRAG` matching the original constructor signature; useful for swapping in without changing call sites.

### Launchers

- `launch_mcp_server.py` — cross-platform Python launcher.
- `start_mcp.ps1` — Windows/PowerShell launcher; supports `-ServerOnly`, `-StreamlitOnly`, configurable `-Config` and `-LogLevel`.

### UI

- `src/ui/streamlit_app_mcp.py` — Streamlit app that talks to the MCP server through `MCPIntegratedRAG`. Maintains a single asyncio loop per session so subprocess pipes stay coherent across Streamlit reruns. Includes a checkbox to toggle MCP off and use direct imports instead.

## Tools, Resources, Prompts

See [MCP_QUICKSTART.md](MCP_QUICKSTART.md) for the full table. In short:

- **Tools (7)**: `chat`, `load_documents`, `search_documents`, `analyze_query`, `get_stats`, `clear_conversation`, `clear_documents`.
- **Resources (6)**: per-role `rag://conversations/{role}`, `rag://config`, `rag://documents/list`.
- **Prompts (1)**: `system_prompt` (role-aware).

## Configuration Touchpoints

- `config.mcp` — protocol version, server timeout, stdio buffer size, transport flags.
- `config.roles[*].mcp_tools` — per-role allowlist of tool names the role may invoke.
- Roles are normalized case-insensitively against `Admin`, `Expert`, `User`, `Guest` before any tool runs.

## Integrating With An MCP Client

Any MCP-compatible client that can spawn a stdio server works. Provide:

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

Place that block under whatever key your client expects (`mcp.servers`, `mcpServers`, etc.).

## Custom Client Example

```python
from src.mcp.client import MCPClient
import asyncio, sys

async def main():
    client = MCPClient([sys.executable, "src/mcp/mcp_server.py"])
    await client.start_server()
    try:
        print(await client.chat("Hello from a custom client!"))
    finally:
        await client.stop_server()

asyncio.run(main())
```

## Verifying the Setup

```bash
python health_check.py
python -m unittest tests.test_system
```

`tests/test_system.py` exercises the initialize + `tools/list` handshake with fake numeric backends, so it runs without Ollama.

## Compatibility With The Direct API

Nothing in `src/core/` was changed for MCP. Existing imports keep working:

```python
from src.core.final_rag_system import FinalRAGChatbot
rag = FinalRAGChatbot(role="User")
rag.chat("Hello")
```

If you want the MCP transport, prefer the async surface:

```python
from src.mcp.client import MCPIntegratedRAG
async with MCPIntegratedRAG(role="User") as rag:
    await rag.chat("Hello")
```
