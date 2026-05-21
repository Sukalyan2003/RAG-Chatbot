"""
Streamlit Frontend for Final RAG Chatbot System

A comprehensive web interface that provides access to all system features including:
- System health monitoring and testing
- Interactive chat interface
- Document management and processing
- Role-based access control
- Configuration management
- Performance analytics
- Data export capabilities
"""

import streamlit as st
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict


# Add src/ to Python path for direct Streamlit execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Configure Streamlit page
st.set_page_config(
    page_title="Final RAG Chatbot System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1e88e5;
    }
    
    .success-card {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    
    .warning-card {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    
    .error-card {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        margin-right: 2rem;
    }
    
    .source-attribution {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGApp:
    """Main Streamlit application class for the RAG Chatbot System."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.initialize_session_state()
        self.run_initial_tests()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # System state
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'system_status' not in st.session_state:
            st.session_state.system_status = {}
        if 'test_results' not in st.session_state:
            st.session_state.test_results = {}
        
        # Chat state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_role' not in st.session_state:
            st.session_state.current_role = "User"
        if 'chatbot_instance' not in st.session_state:
            st.session_state.chatbot_instance = None
        
        # Document state
        if 'loaded_documents' not in st.session_state:
            st.session_state.loaded_documents = []
        if 'document_stats' not in st.session_state:
            st.session_state.document_stats = {}
        
        # Configuration state
        if 'config' not in st.session_state:
            st.session_state.config = self.load_config()
        
        # Analytics state
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = []
    
    def load_config(self) -> Dict:
        """Load system configuration."""
        try:
            with open('config/config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Configuration file not found. Please ensure config/config.json exists.")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Invalid configuration file: {e}")
            return {}
    
    def run_initial_tests(self):
        """Run system tests on application startup."""
        if not st.session_state.get('tests_completed', False):
            with st.spinner("Running initial system tests..."):
                st.session_state.test_results = self.run_comprehensive_tests()
                st.session_state.tests_completed = True
    
    def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive system tests."""
        test_results = {
            'import_test': {'status': 'running', 'details': ''},
            'config_test': {'status': 'running', 'details': ''},
            'dependencies_test': {'status': 'running', 'details': ''},
            'basic_init_test': {'status': 'running', 'details': ''},
            'overall_status': 'running'
        }
        
        # Test 1: Import Tests
        try:
            from core.final_rag_system import FinalRAGChatbot
            from core.document_processor import DocumentProcessor
            from core.embedding_manager import EmbeddingManager
            from core.llm_interface import LLMInterface
            from core.query_analyzer import QueryAnalyzer
            from core.conversation_manager import ConversationManager
            from core.utils import setup_logging, validate_input
            
            test_results['import_test'] = {
                'status': 'passed',
                'details': 'All modules imported successfully'
            }
        except Exception as e:
            test_results['import_test'] = {
                'status': 'failed',
                'details': f'Import error: {str(e)}'
            }
        
        # Test 2: Configuration Test
        try:
            config = self.load_config()
            required_sections = ['system', 'embeddings', 'llm', 'retrieval', 'roles', 'paths']
            missing = [s for s in required_sections if s not in config]
            
            if missing:
                test_results['config_test'] = {
                    'status': 'failed',
                    'details': f'Missing config sections: {missing}'
                }
            else:
                test_results['config_test'] = {
                    'status': 'passed',
                    'details': 'Configuration file is valid'
                }
        except Exception as e:
            test_results['config_test'] = {
                'status': 'failed',
                'details': f'Config error: {str(e)}'
            }
        
        # Test 3: Dependencies Test
        try:
            import sentence_transformers
            import sklearn
            import numpy
            import pandas
            
            test_results['dependencies_test'] = {
                'status': 'passed',
                'details': 'All dependencies available'
            }
        except ImportError as e:
            test_results['dependencies_test'] = {
                'status': 'failed',
                'details': f'Missing dependency: {str(e)}'
            }
        
        # Test 4: Basic Initialization Test
        try:
            if test_results['import_test']['status'] == 'passed':
                from core.final_rag_system import FinalRAGChatbot
                chatbot = FinalRAGChatbot(role="User")
                if chatbot.initialized:
                    test_results['basic_init_test'] = {
                        'status': 'passed',
                        'details': 'System initializes successfully'
                    }
                    st.session_state.system_initialized = True
                else:
                    test_results['basic_init_test'] = {
                        'status': 'failed',
                        'details': 'System failed to initialize'
                    }
            else:
                test_results['basic_init_test'] = {
                    'status': 'skipped',
                    'details': 'Skipped due to import failures'
                }
        except Exception as e:
            test_results['basic_init_test'] = {
                'status': 'failed',
                'details': f'Initialization error: {str(e)}'
            }
        
        # Overall status
        all_tests = [test_results[key]['status'] for key in test_results if key != 'overall_status']
        if all(status == 'passed' for status in all_tests):
            test_results['overall_status'] = 'passed'
        elif any(status == 'failed' for status in all_tests):
            test_results['overall_status'] = 'failed'
        else:
            test_results['overall_status'] = 'partial'
        
        return test_results
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🤖 Final RAG Chatbot System</h1>', unsafe_allow_html=True)
        
        # System status indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            status = st.session_state.test_results.get('overall_status', 'unknown')
            if status == 'passed':
                st.success("✅ System Status: All systems operational")
            elif status == 'failed':
                st.error("❌ System Status: Issues detected")
            elif status == 'partial':
                st.warning("⚠️ System Status: Partial functionality")
            else:
                st.info("🔄 System Status: Checking...")
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("🎛️ Navigation")
        
        # Page selection
        page = st.sidebar.selectbox(
            "Select Page",
            [
                "🏠 Dashboard",
                "💬 Chat Interface", 
                "📚 Document Management",
                "👥 Role Management",
                "⚙️ Configuration",
                "📊 Analytics",
                "🧪 System Tests",
                "📤 Export Center"
            ]
        )
        
        st.sidebar.markdown("---")
        
        # Role selector
        st.sidebar.subheader("👤 Current Role")
        new_role = st.sidebar.selectbox(
            "Role",
            ["User", "Expert", "Admin"],
            index=["User", "Expert", "Admin"].index(st.session_state.current_role)
        )
        
        if new_role != st.session_state.current_role:
            st.session_state.current_role = new_role
            st.session_state.chatbot_instance = None  # Reset chatbot instance
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.subheader("📊 Quick Stats")
        if st.session_state.chatbot_instance:
            try:
                stats = st.session_state.chatbot_instance.get_stats()
                st.sidebar.metric("Queries Processed", stats.get('queries_processed', 0))
                st.sidebar.metric("Success Rate", f"{stats.get('successful_responses', 0)}/{stats.get('queries_processed', 0)}")
                avg_time = stats.get('average_response_time', 0)
                st.sidebar.metric("Avg Response Time", f"{avg_time:.2f}s" if avg_time else "N/A")
            except:
                st.sidebar.info("Stats unavailable")
        else:
            st.sidebar.info("Initialize chatbot to see stats")
        
        return page.split(" ", 1)[1]  # Return page name without emoji
    
    def render_dashboard(self):
        """Render the main dashboard page."""
        st.header("🏠 System Dashboard")
        
        # System overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="status-card {'success-card' if st.session_state.system_initialized else 'error-card'}">
                <h4>System Status</h4>
                <p>{'✅ Operational' if st.session_state.system_initialized else '❌ Issues'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            doc_count = len(st.session_state.loaded_documents)
            st.markdown(f"""
            <div class="status-card">
                <h4>Documents</h4>
                <p>📚 {doc_count} loaded</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            chat_count = len(st.session_state.chat_history)
            st.markdown(f"""
            <div class="status-card">
                <h4>Conversations</h4>
                <p>💬 {chat_count} messages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            role = st.session_state.current_role
            st.markdown(f"""
            <div class="status-card">
                <h4>Current Role</h4>
                <p>👤 {role}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Test results summary
        st.subheader("🧪 System Test Results")
        test_results = st.session_state.test_results
        
        for test_name, result in test_results.items():
            if test_name == 'overall_status':
                continue
                
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'running': '🔄',
                'skipped': '⏭️'
            }.get(result['status'], '❓')
            
            with st.expander(f"{status_icon} {test_name.replace('_', ' ').title()}"):
                st.write(f"**Status:** {result['status']}")
                st.write(f"**Details:** {result['details']}")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Re-run Tests", use_container_width=True):
                with st.spinner("Running tests..."):
                    st.session_state.test_results = self.run_comprehensive_tests()
                    st.rerun()
        
        with col2:
            if st.button("💬 Start Chat", use_container_width=True):
                st.session_state.current_page = "Chat Interface"
                st.rerun()
        
        with col3:
            if st.button("📚 Load Documents", use_container_width=True):
                st.session_state.current_page = "Document Management"
                st.rerun()
    
    def render_chat_interface(self):
        """Render the chat interface page."""
        st.header("💬 Chat Interface")
        
        # Initialize chatbot if needed
        if not st.session_state.chatbot_instance or not st.session_state.system_initialized:
            try:
                from core.final_rag_system import FinalRAGChatbot
                st.session_state.chatbot_instance = FinalRAGChatbot(role=st.session_state.current_role)
                st.success(f"✅ Chatbot initialized for role: {st.session_state.current_role}")
            except Exception as e:
                st.error(f"❌ Failed to initialize chatbot: {e}")
                return
        
        # Chat controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"👤 Chatting as: **{st.session_state.current_role}**")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                if st.session_state.chatbot_instance:
                    st.session_state.chatbot_instance.clear_conversation()
                st.rerun()
        with col3:
            if st.button("📊 Get Stats"):
                if st.session_state.chatbot_instance:
                    stats = st.session_state.chatbot_instance.get_stats()
                    st.json(stats)
        
        # Display chat history
        st.subheader("📝 Conversation History")
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    response_content = message['content']
                    sources = message.get('sources', [])
                    
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant:</strong><br>
                        {response_content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if sources:
                        st.markdown(f"""
                        <div class="source-attribution">
                            📚 Sources: {', '.join(sources[:3])}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Chat input
        st.subheader("✏️ Send Message")
        
        # Message input
        user_input = st.text_area(
            "Your message:",
            height=100,
            placeholder="Ask me anything about your documents..."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            send_button = st.button("📤 Send Message", use_container_width=True)
        with col2:
            stream_mode = st.checkbox("🔄 Streaming", value=False)
        
        # Process message
        if send_button and user_input.strip():
            if not st.session_state.chatbot_instance:
                st.error("❌ Chatbot not initialized. Please check system status.")
                return
            
            # Add user message to history
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Generate response
            with st.spinner("🤖 Generating response..."):
                try:
                    start_time = time.time()
                    response = st.session_state.chatbot_instance.chat(user_input, stream=stream_mode)
                    response_time = time.time() - start_time
                    
                    # Handle different response types
                    if isinstance(response, dict):
                        response_content = response.get('content', str(response))
                        sources = response.get('sources', [])
                    else:
                        response_content = str(response)
                        sources = []
                    
                    # Add bot response to history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response_content,
                        'sources': sources,
                        'response_time': response_time,
                        'timestamp': datetime.now()
                    })
                    
                    # Update performance data
                    st.session_state.performance_data.append({
                        'timestamp': datetime.now(),
                        'response_time': response_time,
                        'role': st.session_state.current_role,
                        'query_length': len(user_input)
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': f"I apologize, but I encountered an error: {str(e)}",
                        'timestamp': datetime.now()
                    })
    
    def render_document_management(self):
        """Render the document management page."""
        st.header("📚 Document Management")
        
        # Document upload section
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'json', 'csv']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            document_type = st.selectbox(
                "Document Type",
                ["auto", "pdf", "txt", "json", "csv", "web"]
            )
        with col2:
            process_button = st.button("🔄 Process Documents", use_container_width=True)
        
        # Process uploaded files
        if process_button and uploaded_files:
            if not st.session_state.chatbot_instance:
                st.error("❌ Please initialize the chatbot first in the Chat Interface.")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded files temporarily
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                saved_files = []
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    saved_files.append(str(file_path))
                    
                    progress = (i + 1) / len(uploaded_files) * 0.5
                    progress_bar.progress(progress)
                    status_text.text(f"Saving {uploaded_file.name}...")
                
                # Process documents
                status_text.text("Processing documents...")
                success = st.session_state.chatbot_instance.load_documents(str(temp_dir), document_type)
                progress_bar.progress(1.0)
                
                if success:
                    st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
                    st.session_state.loaded_documents.extend([f.name for f in uploaded_files])
                    
                    # Update document stats
                    st.session_state.document_stats = {
                        'total_documents': len(st.session_state.loaded_documents),
                        'last_updated': datetime.now(),
                        'types_processed': list(set([f.name.split('.')[-1] for f in uploaded_files]))
                    }
                else:
                    st.error("❌ Failed to process documents. Check logs for details.")
                
                # Cleanup temp files
                import shutil
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {e}")
        
        # Web URL processing
        st.subheader("🌐 Process Web Content")
        col1, col2 = st.columns([3, 1])
        with col1:
            web_url = st.text_input("Enter URL:", placeholder="https://example.com/article")
        with col2:
            web_process_button = st.button("🔄 Process URL")
        
        if web_process_button and web_url:
            if not st.session_state.chatbot_instance:
                st.error("❌ Please initialize the chatbot first.")
                return
            
            with st.spinner("🌐 Processing web content..."):
                try:
                    success = st.session_state.chatbot_instance.load_documents(web_url, "web")
                    if success:
                        st.success(f"✅ Successfully processed content from {web_url}")
                        st.session_state.loaded_documents.append(web_url)
                    else:
                        st.error("❌ Failed to process web content.")
                except Exception as e:
                    st.error(f"❌ Error processing web content: {e}")
        
        # Document library
        st.subheader("📖 Document Library")
        
        if st.session_state.loaded_documents:
            # Display loaded documents
            for i, doc in enumerate(st.session_state.loaded_documents):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {doc}")
                with col2:
                    doc_type = doc.split('.')[-1] if '.' in doc else 'web'
                    st.text(f"Type: {doc_type}")
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.loaded_documents.remove(doc)
                        st.rerun()
        else:
            st.info("📭 No documents loaded yet. Upload some files to get started!")
        
        # Document statistics
        if st.session_state.document_stats:
            st.subheader("📊 Document Statistics")
            stats = st.session_state.document_stats
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Document Types", len(stats.get('types_processed', [])))
            with col3:
                last_updated = stats.get('last_updated')
                if last_updated:
                    st.metric("Last Updated", last_updated.strftime("%H:%M:%S"))
    
    def render_analytics(self):
        """Render the analytics dashboard."""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.performance_data:
            st.info("📊 No performance data available yet. Start chatting to generate analytics!")
            return
        
        # Convert performance data to DataFrame
        df = pd.DataFrame(st.session_state.performance_data)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_response_time = df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col2:
            total_queries = len(df)
            st.metric("Total Queries", total_queries)
        
        with col3:
            if 'role' in df.columns:
                unique_roles = df['role'].nunique()
                st.metric("Active Roles", unique_roles)
            else:
                st.metric("Active Roles", 1)
        
        with col4:
            avg_query_length = df['query_length'].mean() if 'query_length' in df.columns else 0
            st.metric("Avg Query Length", f"{avg_query_length:.0f} chars")
        
        # Response time chart
        st.subheader("⏱️ Response Time Trends")
        
        if len(df) > 1:
            fig_time = px.line(
                df, 
                x='timestamp', 
                y='response_time',
                title='Response Time Over Time',
                labels={'response_time': 'Response Time (seconds)', 'timestamp': 'Time'}
            )
            fig_time.update_layout(xaxis_title="Time", yaxis_title="Response Time (s)")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Need more data points to show trends")
        
        # Role distribution
        if 'role' in df.columns:
            st.subheader("👥 Usage by Role")
            role_counts = df['role'].value_counts()
            
            fig_roles = px.pie(
                values=role_counts.values,
                names=role_counts.index,
                title='Query Distribution by Role'
            )
            st.plotly_chart(fig_roles, use_container_width=True)
        
        # Response time distribution
        st.subheader("📈 Response Time Distribution")
        fig_hist = px.histogram(
            df,
            x='response_time',
            nbins=20,
            title='Response Time Distribution',
            labels={'response_time': 'Response Time (seconds)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Raw data
        with st.expander("📋 Raw Performance Data"):
            st.dataframe(df)
    
    def render_system_tests(self):
        """Render the system tests page."""
        st.header("🧪 System Tests")
        
        # Test controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Run All Tests", use_container_width=True):
                with st.spinner("Running comprehensive tests..."):
                    st.session_state.test_results = self.run_comprehensive_tests()
                    st.rerun()
        
        with col2:
            if st.button("🧹 Clear Test Results", use_container_width=True):
                st.session_state.test_results = {}
                st.rerun()
        
        with col3:
            if st.button("📊 System Info", use_container_width=True):
                st.info(f"""
                **Python Version:** {sys.version}
                **Working Directory:** {os.getcwd()}
                **Streamlit Version:** {st.__version__ if hasattr(st, '__version__') else 'Unknown'}
                """)
        
        # Test results display
        if st.session_state.test_results:
            st.subheader("📋 Test Results")
            
            for test_name, result in st.session_state.test_results.items():
                if test_name == 'overall_status':
                    continue
                
                status = result['status']
                details = result['details']
                
                # Color code based on status
                if status == 'passed':
                    st.success(f"✅ **{test_name.replace('_', ' ').title()}**")
                elif status == 'failed':
                    st.error(f"❌ **{test_name.replace('_', ' ').title()}**")
                elif status == 'running':
                    st.info(f"🔄 **{test_name.replace('_', ' ').title()}**")
                else:
                    st.warning(f"⏭️ **{test_name.replace('_', ' ').title()}**")
                
                st.write(f"**Details:** {details}")
                st.markdown("---")
        
        # Manual test options
        st.subheader("🔧 Manual Tests")
        
        # Test basic chat functionality
        if st.button("💬 Test Basic Chat"):
            try:
                from core.final_rag_system import FinalRAGChatbot
                chatbot = FinalRAGChatbot(role="User")
                response = chatbot.chat("Hello, can you help me?")
                st.success(f"✅ Chat test successful!")
                st.write(f"**Response:** {response}")
            except Exception as e:
                st.error(f"❌ Chat test failed: {e}")
        
        # Test document processing
        if st.button("📄 Test Document Processing"):
            try:
                # Create a simple test document
                test_content = "This is a test document for the RAG system."
                test_file = Path("test_doc.txt")
                test_file.write_text(test_content)
                
                from core.final_rag_system import FinalRAGChatbot
                chatbot = FinalRAGChatbot(role="User")
                success = chatbot.load_documents(str(test_file))
                
                if success:
                    st.success("✅ Document processing test successful!")
                else:
                    st.error("❌ Document processing test failed!")
                
                # Cleanup
                test_file.unlink(missing_ok=True)
                
            except Exception as e:
                st.error(f"❌ Document processing test failed: {e}")
        
        # Dependency checker
        st.subheader("📦 Dependency Check")
        if st.button("🔍 Check Dependencies"):
            dependencies = [
                ("sklearn", "Scikit-learn"),
                ("requests", "Requests"),
                ("bs4", "BeautifulSoup4"),
                ("pdfminer", "PDFMiner"),
                ("numpy", "NumPy"),
                ("pandas", "Pandas"),
                ("plotly", "Plotly")
            ]
            
            for module, name in dependencies:
                try:
                    __import__(module)
                    st.success(f"✅ {name}")
                except ImportError:
                    st.error(f"❌ {name} - Not installed")
    
    def render_export_center(self):
        """Render the export center page."""
        st.header("📤 Export Center")
        
        # Chat history export
        st.subheader("💬 Export Chat History")
        
        if st.session_state.chat_history:
            export_format = st.selectbox("Export Format", ["JSON", "CSV", "TXT"])
            
            if st.button("📥 Export Chat History"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if export_format == "JSON":
                        export_data = json.dumps(st.session_state.chat_history, indent=2, default=str)
                        filename = f"chat_history_{timestamp}.json"
                        
                    elif export_format == "CSV":
                        df = pd.DataFrame(st.session_state.chat_history)
                        export_data = df.to_csv(index=False)
                        filename = f"chat_history_{timestamp}.csv"
                        
                    else:  # TXT
                        lines = []
                        for msg in st.session_state.chat_history:
                            timestamp = msg.get('timestamp', 'Unknown')
                            speaker = "User" if msg['type'] == 'user' else "Assistant"
                            lines.append(f"[{timestamp}] {speaker}: {msg['content']}\n")
                        export_data = "\n".join(lines)
                        filename = f"chat_history_{timestamp}.txt"
                    
                    st.download_button(
                        label=f"💾 Download {export_format} File",
                        data=export_data,
                        file_name=filename,
                        mime="application/octet-stream"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
        else:
            st.info("📭 No chat history to export")
        
        # Performance data export
        st.subheader("📊 Export Performance Data")
        
        if st.session_state.performance_data:
            if st.button("📈 Export Performance Data"):
                try:
                    df = pd.DataFrame(st.session_state.performance_data)
                    csv_data = df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="💾 Download Performance CSV",
                        data=csv_data,
                        file_name=f"performance_data_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
        else:
            st.info("📊 No performance data to export")
        
        # System configuration export
        st.subheader("⚙️ Export System Configuration")
        
        if st.button("🔧 Export Configuration"):
            try:
                config_data = json.dumps(st.session_state.config, indent=2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="💾 Download Configuration JSON",
                    data=config_data,
                    file_name=f"config_backup_{timestamp}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        # System logs export (if available)
        st.subheader("📋 Export System Logs")
        
        logs_dir = Path("data/logs")
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                selected_log = st.selectbox("Select Log File", [f.name for f in log_files])
                
                if st.button("📜 Export Log File"):
                    try:
                        log_path = logs_dir / selected_log
                        log_content = log_path.read_text()
                        
                        st.download_button(
                            label="💾 Download Log File",
                            data=log_content,
                            file_name=selected_log,
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Log export failed: {e}")
            else:
                st.info("📭 No log files found")
        else:
            st.info("📭 Logs directory not found")
    
    def render_configuration(self):
        """Render the configuration management page."""
        st.header("⚙️ Configuration Management")
        
        # Configuration editor
        st.subheader("🔧 Edit Configuration")
        
        # Load current config
        config = st.session_state.config
        
        # System settings
        with st.expander("🖥️ System Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                log_level = st.selectbox(
                    "Log Level",
                    ["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("system", {}).get("log_level", "INFO"))
                )
                
                max_history = st.number_input(
                    "Max Conversation History",
                    min_value=1,
                    max_value=100,
                    value=config.get("system", {}).get("max_conversation_history", 50)
                )
            
            with col2:
                cache_embeddings = st.checkbox(
                    "Cache Embeddings",
                    value=config.get("system", {}).get("cache_embeddings", True)
                )
                
                enable_streaming = st.checkbox(
                    "Enable Streaming",
                    value=config.get("system", {}).get("enable_streaming", False)
                )
        
        # LLM settings
        with st.expander("🤖 LLM Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                provider_options = ["ollama", "openai", "local", "custom"]
                current_provider = config.get("llm", {}).get("provider", "ollama")
                if current_provider not in provider_options:
                    current_provider = "custom"
                llm_provider = st.selectbox(
                    "LLM Provider",
                    provider_options,
                    index=provider_options.index(current_provider)
                )
                
                llm_model = st.text_input(
                    "Model Name",
                    value=config.get("llm", {}).get("model", "qwen3:4b-instruct")
                )
            
            with col2:
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=4000,
                    value=config.get("llm", {}).get("max_tokens", 1500)
                )
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=config.get("llm", {}).get("temperature", 0.7),
                    step=0.1
                )
        
        # Embedding settings
        with st.expander("🔤 Embedding Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                embedding_provider_options = ["ollama", "sentence_transformers"]
                current_embedding_provider = config.get("embedding", {}).get("provider", "ollama")
                if current_embedding_provider not in embedding_provider_options:
                    current_embedding_provider = "ollama"
                embedding_provider = st.selectbox(
                    "Embedding Provider",
                    embedding_provider_options,
                    index=embedding_provider_options.index(current_embedding_provider)
                )

                embedding_model = st.text_input(
                    "Embedding Model",
                    value=config.get("embedding", {}).get("model", "qwen3-embedding:0.6b")
                )
                
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=128,
                    value=config.get("embedding", {}).get("batch_size", 32)
                )
            
            with col2:
                device = st.selectbox(
                    "Device",
                    ["cpu", "cuda", "auto"],
                    index=["cpu", "cuda", "auto"].index(config.get("embedding", {}).get("device", "cpu"))
                )
                
                max_length = st.number_input(
                    "Max Sequence Length",
                    min_value=128,
                    max_value=1024,
                    value=config.get("embedding", {}).get("max_length", 512)
                )
        
        # Retrieval settings
        with st.expander("🔍 Retrieval Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_results = st.number_input(
                    "Max Results",
                    min_value=1,
                    max_value=20,
                    value=config.get("retrieval", {}).get("max_results", 5)
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.get("retrieval", {}).get("similarity_threshold", 0.3),
                    step=0.05
                )
            
            with col2:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=config.get("retrieval", {}).get("chunk_size", 500)
                )
                
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=200,
                    value=config.get("retrieval", {}).get("chunk_overlap", 50)
                )
        
        # Save configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Configuration", use_container_width=True):
                # Update configuration
                new_config = dict(config)
                new_config["system"] = {
                    **config.get("system", {}),
                    "log_level": log_level,
                    "max_conversation_history": max_history,
                    "cache_embeddings": cache_embeddings,
                    "enable_streaming": enable_streaming
                }
                new_config["embedding"] = {
                    **config.get("embedding", {}),
                    "provider": embedding_provider,
                    "model": embedding_model,
                    "device": device,
                    "batch_size": batch_size,
                    "max_length": max_length
                }
                new_config["llm"] = {
                    **config.get("llm", {}),
                    "provider": llm_provider,
                    "model": llm_model,
                    "api_key": config.get("llm", {}).get("api_key", ""),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "base_url": config.get("llm", {}).get("base_url", "")
                }
                new_config["retrieval"] = {
                    **config.get("retrieval", {}),
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
                
                try:
                    with open("config/config.json", "w") as f:
                        json.dump(new_config, f, indent=2)
                    
                    st.session_state.config = new_config
                    st.success("✅ Configuration saved successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Failed to save configuration: {e}")
        
        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                if st.session_state.get('confirm_reset', False):
                    # Reset logic here
                    st.info("Reset functionality would be implemented here")
                    st.session_state.confirm_reset = False
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ Click again to confirm reset")
        
        with col3:
            if st.button("📤 Export Config", use_container_width=True):
                config_json = json.dumps(st.session_state.config, indent=2)
                st.download_button(
                    "💾 Download Config",
                    config_json,
                    "config_export.json",
                    "application/json"
                )
    
    def run(self):
        """Main application entry point."""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get current page
            current_page = self.render_sidebar()
            
            # Render main content based on selected page
            if current_page == "Dashboard":
                self.render_dashboard()
            elif current_page == "Chat Interface":
                self.render_chat_interface()
            elif current_page == "Document Management":
                self.render_document_management()
            elif current_page == "Role Management":
                st.header("👥 Role Management")
                st.info("🚧 Role management features coming soon!")
            elif current_page == "Configuration":
                self.render_configuration()
            elif current_page == "Analytics":
                self.render_analytics()
            elif current_page == "System Tests":
                self.render_system_tests()
            elif current_page == "Export Center":
                self.render_export_center()
            else:
                st.error(f"Unknown page: {current_page}")
            
        except Exception as e:
            st.error(f"❌ Application error: {e}")
            st.exception(e)

# Main execution
if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()
