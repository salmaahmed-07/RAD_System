# app.py
import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Import your existing modules
from rag import RAGWithCitations
from document_processor import DocumentProcessor
from citation_utils import format_citations_html, format_citations_simple, print_citation_summary

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Telecom Egypt Intelligent Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .header h1 {
        margin: 0;
        font-size: 28px;
    }
    
    .header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }
    
    .assistant-message {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #43a047;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Citations */
    .citation-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 4px solid #ff9800;
    }
    
    .citation-box h4 {
        margin-top: 0;
        color: #e65100;
    }
    
    .citation-item {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .sidebar-section {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stat-box {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        text-align: center;
    }
    
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #1a237e;
    }
    
    .stat-label {
        font-size: 12px;
        color: #666;
    }
    
    /* Upload section */
    .upload-section {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Initialize Session State
# ==========================================

def initialize_session_state():
    """Initialize all session state variables"""
    
    if 'rag_system' not in st.session_state:
        # Load chunks
        with open("embeddings.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # Initialize RAG system
        from rag import RAGWithCitations
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        st.session_state.rag_system = RAGWithCitations(chunks, model)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'total_chunks' not in st.session_state:
        with open("embeddings.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        st.session_state.total_chunks = len(chunks)

# ==========================================
# Sidebar
# ==========================================

def render_sidebar():
    """Render the sidebar with controls and stats"""
    
    with st.sidebar:
        # Header
        st.image("https://te.eg/favicon.ico", width=50) if os.path.exists("logo.png") else st.write("📡")
        st.title("📡 Telecom Egypt")
        st.markdown("---")
        
        # Document Upload Section
        st.subheader("📤 Upload Documents")
        st.markdown("Upload documents to enhance the knowledge base")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt', 'html'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📁 {len(uploaded_files)} file(s) selected")
            with col2:
                if st.button("📥 Process", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        for file in uploaded_files:
                            try:
                                chunks = st.session_state.processor.process_file(file)
                                st.session_state.rag_system.chunks.extend(chunks)
                                # Save to file
                                st.session_state.processor.save_chunks(chunks, "embeddings.json")
                                st.session_state.total_chunks += len(chunks)
                                st.success(f"✅ {file.name}")
                            except Exception as e:
                                st.error(f"❌ {file.name}: {str(e)}")
                    st.session_state.documents_loaded = True
                    st.rerun()
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{st.session_state.total_chunks}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "🟢 Online" if st.session_state.total_chunks > 0 else "🟡 No Data"
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value" style="font-size: 18px;">{status}</div>
                <div class="stat-label">System Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat stats
        total_messages = len(st.session_state.chat_history)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{total_messages}</div>
            <div class="stat-label">Total Messages</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Actions
        st.subheader("⚙️ Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.caption("🔒 All data is processed locally")
        st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ==========================================
# Main Chat Interface
# ==========================================

def render_chat():
    """Render the main chat interface"""
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>💬 Telecom Egypt Intelligent Assistant</h1>
        <p>Ask questions about Telecom Egypt services, plans, and more</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                
                # Display citations if available
                if "citations" in message and message["citations"]:
                    st.markdown("""
                    <div class="citation-box">
                        <h4>📚 Sources Used</h4>
                    """, unsafe_allow_html=True)
                    
                    for citation in message["citations"]:
                        st.markdown(f"""
                        <div class="citation-item">
                            <strong>[{citation['id']}]</strong>
                            <span style="color: #e65100; font-weight: bold;">{citation['source']}</span>
                            <span style="color: #666; font-size: 0.9em;">(Relevance: {citation['relevance_score']:.4f})</span>
                            <br>
                            <span style="color: #333; font-size: 0.95em;">{citation['text'][:200]}...</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    with col1:
        user_query = st.chat_input("Ask about Telecom Egypt services...")
    with col2:
        # Language toggle could go here
        pass
    
    if user_query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get response
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.rag_system.query(user_query, k=5)
        
        # Add assistant response with citations
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "citations": result.get("citations", []),
            "all_citations": result.get("all_citations", []),
            "timestamp": datetime.now().isoformat()
        })
        
        st.rerun()

# ==========================================
# Main
# ==========================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat()

if __name__ == "__main__":
    main()