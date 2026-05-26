"""
MCP-Enabled Streamlit Application for Final RAG Chatbot

This is an enhanced version of the Streamlit interface that leverages
Model Context Protocol (MCP) for all interactions with the RAG system.
"""

import streamlit as st
import asyncio
import importlib.util
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Add src/ to the front of Python path so the local mcp package wins over the
# third-party package named "mcp".
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MCP-enabled components
from mcp.client import MCPIntegratedRAG
from core.utils import validate_input

# Configure Streamlit
st.set_page_config(
    page_title="Final RAG Chatbot (MCP)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _load_config_safe() -> Dict[str, Any]:
    """Read ``config/config.json`` once per call without raising."""
    try:
        with open("config/config.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _streaming_enabled_in_config() -> bool:
    """Return ``system.enable_streaming`` from ``config/config.json``.

    Read on demand so a config edit takes effect on the next chat turn
    without restarting Streamlit.
    """
    return bool(_load_config_safe().get("system", {}).get("enable_streaming", False))


def _mcp_enabled_in_config() -> bool:
    """Return ``mcp.enabled`` from ``config/config.json`` (default True)."""
    return bool(_load_config_safe().get("mcp", {}).get("enabled", True))


def init_session_state():
    """Initialize session state variables."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "async_loop" not in st.session_state:
        st.session_state.async_loop = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_role" not in st.session_state:
        st.session_state.current_role = "User"
    if "mcp_enabled" not in st.session_state:
        st.session_state.mcp_enabled = _mcp_enabled_in_config()
    if "system_stats" not in st.session_state:
        st.session_state.system_stats = {}

def get_session_loop() -> asyncio.AbstractEventLoop:
    """Use one event loop per Streamlit session so MCP subprocess futures stay on the same loop."""
    loop = st.session_state.get("async_loop")
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state.async_loop = loop
    return loop

def run_async(coro):
    """Run async MCP work on the session loop instead of creating a new loop per click."""
    loop = get_session_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def get_rag_system(role: str = None) -> MCPIntegratedRAG:
    """Get or create RAG system instance."""
    if role is None:
        role = st.session_state.current_role
    
    mode_changed = (
        st.session_state.rag_system is not None
        and st.session_state.rag_system.use_mcp != st.session_state.mcp_enabled
    )

    if mode_changed:
        try:
            await st.session_state.rag_system.__aexit__(None, None, None)
        except Exception:
            pass
        st.session_state.rag_system = None

    if st.session_state.rag_system is None:
        # Create new instance
        st.session_state.rag_system = MCPIntegratedRAG(
            role=role,
            config_path="config/config.json",
            use_mcp=st.session_state.mcp_enabled
        )
        await st.session_state.rag_system.__aenter__()
    else:
        st.session_state.rag_system.role = role
    
    return st.session_state.rag_system

def sidebar():
    """Create sidebar with controls."""
    st.sidebar.title("🤖 RAG Chatbot Settings")
    
    # Role selection
    new_role = st.sidebar.selectbox(
        "Select Role",
        ["User", "Expert", "Admin"],
        index=["User", "Expert", "Admin"].index(st.session_state.current_role)
    )
    
    if new_role != st.session_state.current_role:
        st.session_state.current_role = new_role
        if st.session_state.rag_system:
            st.session_state.rag_system.role = new_role
        st.rerun()
    
    # MCP toggle
    st.session_state.mcp_enabled = st.sidebar.checkbox(
        "Use MCP Protocol",
        value=st.session_state.mcp_enabled,
        help="Enable Model Context Protocol for standardized communication"
    )
    
    st.sidebar.markdown("---")
    
    # Document management
    st.sidebar.subheader("📄 Document Management")
    
    upload_types = ["pdf", "txt", "md", "json", "csv"]
    if importlib.util.find_spec("docx"):
        upload_types.append("docx")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Document",
        type=upload_types,
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_file:
        if st.sidebar.button("Process Document"):
            process_uploaded_document(uploaded_file)
    
    document_dir = st.sidebar.text_input(
        "Document Directory",
        placeholder="Enter path to document directory"
    )
    
    if document_dir and st.sidebar.button("Load Directory"):
        process_uploaded_directory(document_dir)
    
    st.sidebar.markdown("---")
    
    # System controls
    st.sidebar.subheader("⚙️ System Controls")
    
    if st.sidebar.button("Clear Conversation"):
        run_async(clear_conversation())

    if st.sidebar.button("Clear Documents / Cache"):
        run_async(clear_documents())
    
    if st.sidebar.button("Refresh Stats"):
        run_async(update_system_stats())
    
    if st.sidebar.button("Export Chat"):
        export_chat_history()

INGEST_STAGE_LABELS = {
    "reading": "📖 Reading files",
    "chunking": "✂️ Chunking text",
    "dedup": "♻️ Deduplicating chunks",
    "embedding": "🧬 Embedding chunks",
    "storing": "🗄️ Storing embeddings",
    "saving_cache": "💾 Persisting cache",
}

def render_response_details(metadata: Dict[str, Any]) -> None:
    """Render assistant Response Details with retrieved chunks surfaced inline.

    The retrieved-chunks block is what tells you whether the answer was actually
    grounded in your documents. Each chunk is shown as its own expander titled
    with the source and similarity score; the content is rendered as markdown
    so highlighted phrases in the answer are easy to scan against.
    """
    if not metadata:
        return

    chunks = metadata.get("retrieved_chunks") or []
    count = metadata.get("search_results_count")
    if count is None:
        count = len(chunks)

    st.markdown(f"**Retrieved chunks ({count})** — use these to verify grounding.")
    if not chunks:
        st.warning(
            "No chunks were retrieved for this query. The answer could not have been grounded "
            "in your documents — anything the model said came from its training data."
        )
    else:
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            score = chunk.get("score")
            score_str = f"score {score:.3f}" if isinstance(score, (int, float)) else "score n/a"
            with st.expander(f"[{idx}] {Path(str(source)).name} · {score_str}"):
                content = chunk.get("content") or ""
                st.markdown(f"> {content}" if content else "_(empty chunk)_")
                meta_payload = chunk.get("metadata") or {}
                if meta_payload:
                    st.caption(f"Source path: `{source}`")
                    st.json(meta_payload)

    other_meta = {
        key: value
        for key, value in metadata.items()
        if key not in {"retrieved_chunks", "search_results_count"}
    }
    if other_meta:
        with st.expander("Query analysis / debug"):
            st.json(other_meta)


CHAT_STAGE_LABELS = {
    "validating": "🔒 Validating input",
    "analyzing": "🧠 Analyzing query",
    "retrieving": "📚 Retrieving relevant context",
    "context": "🧵 Loading conversation context",
    "generating": "✍️ Generating answer",
    "done": "✅ Done",
}


def process_uploaded_document(uploaded_file):
    """Process an uploaded document using real MCP progress notifications."""
    try:
        # Save uploads under data/documents so cached sources remain valid after restart.
        upload_dir = Path("data/documents/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(uploaded_file.name).name
        safe_name = "".join(
            char if char.isalnum() or char in "._- " else "_"
            for char in safe_name
        )
        file_path = upload_dir / safe_name

        import time as _time
        start = _time.time()
        with st.sidebar.status(f"⏳ Ingesting {safe_name}…", expanded=True) as ingest_status:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            ingest_status.write("📥 Wrote upload to disk")

            progress_bar = st.sidebar.progress(0.0, text="Waiting for server")
            success = False
            message = ""
            last_stage = ""

            def on_progress(stage, current, total):
                nonlocal last_stage
                label = INGEST_STAGE_LABELS.get(stage, stage or "Working")
                if stage == "embedding" and total:
                    ratio = (current or 0) / total if total else 0
                    progress_bar.progress(
                        min(ratio, 1.0),
                        text=f"{label}: {current or 0}/{total} chunks",
                    )
                elif stage in ("dedup", "storing") and total:
                    ratio = (current or 0) / total if total else 0
                    progress_bar.progress(min(ratio, 1.0), text=label)
                if stage != last_stage:
                    ingest_status.write(label)
                    last_stage = stage
                elapsed = _time.time() - start
                ingest_status.update(label=f"{label} · {elapsed:.1f}s")

            async def run_ingest():
                nonlocal success, message
                success, message = await load_document_file(str(file_path), on_progress)

            run_async(run_ingest())
            elapsed = _time.time() - start
            progress_bar.progress(1.0 if success else 0.0, text="Done" if success else "Failed")
            ingest_status.update(
                label=f"{'✅ Ingest complete' if success else '❌ Ingest failed'} · {elapsed:.1f}s",
                state="complete" if success else "error",
            )

        if success:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    except Exception as e:
        st.sidebar.error(f"Error processing document: {e}")

async def load_document_file(file_path: str, progress_callback=None):
    """Load a single document file, forwarding MCP progress notifications."""
    try:
        rag_system = await get_rag_system("Admin")  # Use Admin role for document loading
        success = await rag_system.load_documents(file_path, progress_callback=progress_callback)
        message = rag_system.last_operation_message or (
            f"Loaded {Path(file_path).name}" if success else f"Failed to load {Path(file_path).name}"
        )
        return success, message
    except Exception as e:
        return False, f"Error loading {Path(file_path).name}: {e}"

async def load_document_directory(directory: str, progress_callback=None):
    """Load documents from a directory, forwarding MCP progress notifications."""
    try:
        rag_system = await get_rag_system("Admin")  # Use Admin role for document loading
        success = await rag_system.load_documents(directory, progress_callback=progress_callback)
        message = rag_system.last_operation_message or (
            f"Loaded documents from: {directory}"
            if success
            else f"Failed to load documents from: {directory}"
        )
        return success, message
    except Exception as e:
        return False, f"Error loading directory: {e}"


def process_uploaded_directory(directory: str):
    """Load a directory with real MCP progress notifications."""
    try:
        import time as _time
        start = _time.time()
        with st.sidebar.status(f"⏳ Ingesting {directory}…", expanded=True) as ingest_status:
            progress_bar = st.sidebar.progress(0.0, text="Waiting for server")
            success = False
            message = ""
            last_stage = ""

            def on_progress(stage, current, total):
                nonlocal last_stage
                label = INGEST_STAGE_LABELS.get(stage, stage or "Working")
                if stage == "embedding" and total:
                    ratio = (current or 0) / total if total else 0
                    progress_bar.progress(
                        min(ratio, 1.0),
                        text=f"{label}: {current or 0}/{total} chunks",
                    )
                elif stage in ("dedup", "storing") and total:
                    ratio = (current or 0) / total if total else 0
                    progress_bar.progress(min(ratio, 1.0), text=label)
                if stage != last_stage:
                    ingest_status.write(label)
                    last_stage = stage
                elapsed = _time.time() - start
                ingest_status.update(label=f"{label} · {elapsed:.1f}s")

            async def run_ingest():
                nonlocal success, message
                success, message = await load_document_directory(directory, on_progress)

            run_async(run_ingest())
            elapsed = _time.time() - start
            progress_bar.progress(1.0 if success else 0.0, text="Done" if success else "Failed")
            ingest_status.update(
                label=f"{'✅ Directory ingested' if success else '❌ Ingest failed'} · {elapsed:.1f}s",
                state="complete" if success else "error",
            )

        if success:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    except Exception as e:
        st.sidebar.error(f"Error loading directory: {e}")

async def clear_conversation():
    """Clear conversation history."""
    try:
        rag_system = await get_rag_system()
        await rag_system.clear_conversation()
        st.session_state.chat_history = []
        st.success("Conversation cleared!")
    except Exception as e:
        st.error(f"Error clearing conversation: {e}")

async def clear_documents():
    """Clear loaded documents and the persisted embeddings cache."""
    try:
        rag_system = await get_rag_system("Admin")
        message = await rag_system.clear_documents(delete_cache=True)
        st.session_state.document_list = []
        st.session_state.system_stats = {}
        st.success(f"Documents/cache cleared: {message}")
    except Exception as e:
        st.error(f"Error clearing documents/cache: {e}")

async def update_system_stats():
    """Update system statistics."""
    try:
        rag_system = await get_rag_system()
        stats = await rag_system.get_stats()
        st.session_state.system_stats = stats
    except Exception as e:
        st.error(f"Error updating stats: {e}")

def export_chat_history():
    """Export chat history."""
    try:
        if st.session_state.chat_history:
            # Convert to DataFrame for easy export
            df = pd.DataFrame(st.session_state.chat_history)
            csv = df.to_csv(index=False)
            
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.warning("No chat history to export")
    except Exception as e:
        st.sidebar.error(f"Error exporting chat: {e}")

def main_interface():
    """Main chat interface."""
    st.title("🤖 Final RAG Chatbot")
    st.markdown(f"**Current Role:** {st.session_state.current_role} | **MCP Enabled:** {st.session_state.mcp_enabled}")
    
    # Display system status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queries Processed", st.session_state.system_stats.get("queries_processed", 0))
    
    with col2:
        avg_time = st.session_state.system_stats.get("average_response_time", 0)
        st.metric("Avg Response Time", f"{avg_time:.2f}s" if avg_time > 0 else "N/A")
    
    with col3:
        success_rate = 0
        total = st.session_state.system_stats.get("queries_processed", 0)
        if total > 0:
            successful = st.session_state.system_stats.get("successful_responses", 0)
            success_rate = (successful / total) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        errors = st.session_state.system_stats.get("errors", 0)
        st.metric("Errors", errors)
    
    st.markdown("---")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("metadata"):
                    with st.expander("Response Details"):
                        render_response_details(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response — real MCP progress notifications drive the status.
        # The status panel sits above the answer; the answer (streamed or
        # one-shot) is rendered as its own block in the chat bubble so it
        # never gets nested inside a collapsing status.
        with st.chat_message("assistant"):
            import time as _time
            start = _time.time()
            chat_status = st.status("🤖 Thinking…", expanded=True)
            response = None
            last_stage = ""

            def on_progress(stage, current, total):
                nonlocal last_stage
                label = CHAT_STAGE_LABELS.get(stage, stage or "Working")
                if stage == "retrieving" and current is not None:
                    label = f"{label} ({current}/{total or '?'} chunks)"
                if stage != last_stage:
                    chat_status.write(label)
                    last_stage = stage
                elapsed = _time.time() - start
                chat_status.update(label=f"{label} · {elapsed:.1f}s")

            use_stream = (
                _streaming_enabled_in_config()
                and not st.session_state.mcp_enabled
            )

            if use_stream:
                # Direct-import path: stream tokens straight into the chat
                # bubble (below the status block, not inside it).
                rag_system = run_async(get_rag_system())
                streamed_text = st.write_stream(
                    rag_system.chat_stream(prompt, on_progress)
                )
                response_time = _time.time() - start
                chat_status.update(
                    label=f"✅ Answer ready · {response_time:.1f}s",
                    state="complete",
                )

                # After the stream finishes, fetch the retrieved chunks and
                # query analysis so Response Details mirrors the non-stream
                # path. Failures here are non-fatal; we still show the
                # answer.
                metadata = {
                    "role": st.session_state.current_role,
                    "mcp_enabled": st.session_state.mcp_enabled,
                    "streamed": True,
                }
                try:
                    async def _collect_meta():
                        analysis = await rag_system.analyze_query(prompt)
                        chunks = await rag_system.search_documents(prompt, max_results=3)
                        return analysis, chunks

                    query_analysis, search_results = run_async(_collect_meta())
                    metadata.update(
                        {
                            "query_analysis": query_analysis,
                            "search_results_count": len(search_results),
                            "retrieved_chunks": search_results,
                        }
                    )
                except Exception as meta_exc:
                    metadata["metadata_error"] = str(meta_exc)

                response = {
                    "role": "assistant",
                    "content": streamed_text,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata,
                }
            else:
                async def run_with_progress():
                    nonlocal response
                    response = await generate_response(prompt, on_progress)

                run_async(run_with_progress())
                response_time = _time.time() - start
                chat_status.update(
                    label=f"✅ Answer ready · {response_time:.1f}s",
                    state="complete",
                )
                st.markdown(response["content"])

            # Show metadata if available, with retrieved chunks surfaced inline
            if response.get("metadata"):
                with st.expander("Response Details", expanded=True):
                    render_response_details(response["metadata"])
        
        # Add assistant response to chat history
        st.session_state.chat_history.append(response)

async def generate_response(prompt: str, progress_callback=None) -> Dict[str, Any]:
    """Generate response using the RAG system."""
    try:
        rag_system = await get_rag_system()

        # Validate input
        if not validate_input(prompt, {}):  # Use empty config for basic validation
            return {
                "role": "assistant",
                "content": "Sorry, your input contains invalid content. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

        # Generate response — progress notifications drive the UI status
        response_text = await rag_system.chat(prompt, progress_callback=progress_callback)
        
        # Get additional metadata
        try:
            query_analysis = await rag_system.analyze_query(prompt)
            search_results = await rag_system.search_documents(prompt, max_results=3)

            metadata = {
                "query_analysis": query_analysis,
                "search_results_count": len(search_results),
                "retrieved_chunks": search_results,
                "role": st.session_state.current_role,
                "mcp_enabled": st.session_state.mcp_enabled,
            }
        except Exception as meta_exc:
            metadata = {
                "role": st.session_state.current_role,
                "mcp_enabled": st.session_state.mcp_enabled,
                "metadata_error": str(meta_exc),
            }
        
        return {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Error generating response: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"error": True}
        }

def analytics_tab():
    """Analytics and monitoring tab."""
    st.header("📊 System Analytics")
    
    if st.button("Refresh Analytics"):
        run_async(update_system_stats())
    
    if st.session_state.system_stats:
        # Display detailed stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            stats_df = pd.DataFrame(
                [
                    {"Metric": key, "Value": json.dumps(value) if isinstance(value, (dict, list)) else str(value)}
                    for key, value in st.session_state.system_stats.items()
                ]
            )
            st.dataframe(stats_df, width="stretch", hide_index=True)
        
        with col2:
            st.subheader("Usage Overview")
            
            # Create some basic charts if we have data
            queries = st.session_state.system_stats.get("queries_processed", 0)
            errors = st.session_state.system_stats.get("errors", 0)
            successful = st.session_state.system_stats.get("successful_responses", 0)
            
            if queries > 0:
                chart_data = pd.DataFrame({
                    "Metric": ["Successful", "Errors"],
                    "Count": [successful, errors]
                })
                st.bar_chart(chart_data.set_index("Metric"))
    else:
        st.info("No analytics data available. Start chatting to generate statistics.")

def documents_tab():
    """Document management tab."""
    st.header("📄 Document Management")
    
    if st.button("Refresh Document List"):
        run_async(load_document_list())
    
    # Display loaded documents (if any stored in session state)
    if "document_list" in st.session_state and st.session_state.document_list:
        st.subheader("Loaded Documents")
        
        docs_df = pd.DataFrame(st.session_state.document_list)
        display_columns = [column for column in ["document", "type", "chunks", "source"] if column in docs_df.columns]
        docs_df = docs_df[display_columns]
        docs_df = docs_df.rename(columns={
            "document": "Document",
            "type": "Type",
            "chunks": "Chunks",
            "source": "Source",
        })
        st.dataframe(docs_df, width="stretch", hide_index=True)
    else:
        st.info("No documents loaded. Use the sidebar to upload or load documents.")

async def load_document_list():
    """Load the list of documents."""
    try:
        rag_system = await get_rag_system("Admin")
        document_list = await rag_system.get_document_list()
        st.session_state.document_list = document_list
    except Exception as e:
        st.error(f"Error loading document list: {e}")

def main():
    """Main application."""
    init_session_state()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "📄 Documents"])
    
    # Sidebar
    sidebar()
    
    with tab1:
        main_interface()
    
    with tab2:
        analytics_tab()
    
    with tab3:
        documents_tab()
    
    # Update stats periodically
    if "last_stats_update" not in st.session_state:
        st.session_state.last_stats_update = datetime.now()
        run_async(update_system_stats())

if __name__ == "__main__":
    main()
