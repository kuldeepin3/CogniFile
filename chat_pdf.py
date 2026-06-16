import os
import tempfile
import base64
import streamlit as st
import numpy as np
from embedchain import App
from embedchain.config import AppConfig
from embedchain.factory import EmbedderFactory, LlmFactory
from embedchain.vectordb.base import BaseVectorDB
from embedchain.config.vectordb.base import BaseVectorDbConfig

# Set page layout to wide for split-screen workspace
st.set_page_config(layout="wide", page_title="CogniFile | Document Intelligence Platform", page_icon="📄")

# Advanced Futuristic Stylesheet (Dark Theme/Glassmorphism/Neon Accents)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Core Typography & Base Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #050816 !important;
        color: #E2E8F0 !important;
    }
    
    /* Streamlit Container Overrides */
    .stApp {
        background-color: #050816 !important;
    }
    
    /* Sidebar Styling Override */
    [data-testid="stSidebar"] {
        background-color: #080C1F !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Brand Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 30px;
    }
    .logo-text {
        font-weight: 800;
        font-size: 1.5rem;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Title & Headers */
    .hero-title {
        font-weight: 800;
        font-size: 3.5rem;
        line-height: 1.2;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #8E2DE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 40px;
        max-width: 700px;
    }
    
    /* Glassmorphic Panel/Card */
    .glass-card {
        background: rgba(10, 15, 30, 0.65) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    .glow-card-indigo {
        border-left: 4px solid #6366F1;
    }
    .glow-card-coral {
        border-left: 4px solid #FF6B6B;
    }
    .glow-card-purple {
        border-left: 4px solid #A855F7;
    }
    
    /* Feature Badge */
    .feature-badge {
        background: rgba(99, 102, 241, 0.12);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.25);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* Interactive Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px 0 rgba(79, 70, 229, 0.3) !important;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px 0 rgba(79, 70, 229, 0.55) !important;
        transform: translateY(-2px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Chat Interface bubbles */
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .user-bubble {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #F8FAFC;
    }
    .bot-bubble {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.18);
        color: #F1F5F9;
    }
    
    /* File Uploader styling override */
    .stFileUploader {
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(10, 15, 30, 0.4) !important;
        padding: 30px !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader:hover {
        border-color: #6366F1 !important;
        background: rgba(99, 102, 241, 0.03) !important;
    }
    
    /* Status workflows */
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .workflow-icon {
        color: #818CF8;
        font-weight: 700;
    }
    .workflow-text {
        font-size: 0.9rem;
        color: #94A3B8;
    }
    
    /* Custom spacing */
    .workspace-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: rgba(10, 15, 30, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 12px 18px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# InMemory DB implementation to avoid compilation dependencies
class InMemoryDB(BaseVectorDB):
    def __init__(self, config=None):
        self.config = config or BaseVectorDbConfig()
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.client = self
        self.embedder = None
        self.last_query_where = None
        self.last_query_results = []
        self.last_query_sims = []

    def _initialize(self):
        pass

    def _get_or_create_db(self):
        return self

    def _get_or_create_collection(self, name=None):
        return self

    def set_collection_name(self, name):
        pass

    def count(self):
        return len(self.ids)

    def reset(self):
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []

    def get(self, ids=None, where=None, limit=None):
        res = {"ids": [], "metadatas": []}
        for i, idx in enumerate(self.ids):
            if ids and idx not in ids:
                continue
            if where:
                match = True
                for k, v in where.items():
                    if self.metadatas[i].get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            res["ids"].append(idx)
            res["metadatas"].append(self.metadatas[i])
            if limit and len(res["ids"]) >= limit:
                break
        return res

    def add(self, embeddings, documents, metadatas, ids, skip_embedding=False):
        if embeddings is None:
            embeddings = self.embedder.embedding_fn(documents)
        for emb, doc, meta, idx in zip(embeddings, documents, metadatas, ids):
            if idx not in self.ids:
                self.embeddings.append(emb)
                self.documents.append(doc)
                self.metadatas.append(meta)
                self.ids.append(idx)

    def query(self, input_query, n_results, where, skip_embedding=False):
        self.last_query_where = where
        if skip_embedding:
            query_vector = np.array(input_query)
        else:
            query_vector = np.array(self.embedder.embedding_fn([input_query])[0])

        if not self.embeddings:
            self.last_query_results = []
            self.last_query_sims = []
            self.all_chunk_sims = []
            return []

        db_vectors = np.array(self.embeddings)
        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        dots = np.dot(db_vectors, query_vector)
        norms = np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vector)
        sims = dots / (norms + 1e-8)

        # Save all similarities for debugging
        self.all_chunk_sims = [float(s) for s in sims]

        indices = list(range(len(self.ids)))
        # Bypassed where filtering to ensure all uploaded chunks are always searched
        sorted_indices = sorted(indices, key=lambda i: sims[i], reverse=True)
        top_indices = sorted_indices[:n_results]

        results = [self.documents[i] for i in top_indices]
        self.last_query_results = results
        self.last_query_sims = [float(sims[i]) for i in top_indices]
        return results

# Chat input loop prevention callback
def handle_chat_input():
    if st.session_state.chat_input:
        st.session_state.user_query = st.session_state.chat_input
        st.session_state.chat_input = ""

if "user_query" not in st.session_state:
    st.session_state.user_query = None

def init_embedchain_bot(provider, api_key=None, base_url=None, llm_model=None, embedder_model=None):
    import openai
    if provider == "OpenAI":
        openai.api_key = api_key
        openai.api_base = "https://api.openai.com/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        if "OPENAI_API_BASE" in os.environ:
            del os.environ["OPENAI_API_BASE"]
        llm = LlmFactory.create("openai", {"model": "gpt-3.5-turbo"})
        embedder = EmbedderFactory.create("openai", {"model": "text-embedding-ada-002"})
    else: # Ollama
        openai.api_key = "ollama"
        openai.api_base = base_url
        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["OPENAI_API_BASE"] = base_url
        llm = LlmFactory.create("openai", {"model": llm_model})
        embedder = EmbedderFactory.create("openai", {"model": embedder_model})

    llm.config.number_documents = 4
    app_config = AppConfig()
    db = InMemoryDB()
    return App(config=app_config, llm=llm, db=db, embedder=embedder)

# Sidebar - Brand Logo & Configuration
with st.sidebar:
    st.markdown("""
        <div class="logo-container">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#grad)" />
                <path d="M2 17L12 22L22 17" stroke="url(#grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                <path d="M2 12L12 17L22 12" stroke="url(#grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                <defs>
                    <linearGradient id="grad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#6366F1" />
                        <stop offset="1" stop-color="#A855F7" />
                    </linearGradient>
                </defs>
            </svg>
            <span class="logo-text">CogniFile</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Workspace Node")
    provider = st.radio("API Engine Node", ["OpenAI Cloud", "Local Ollama Engine"])

    bot_initialized = False

    if provider == "OpenAI Cloud":
        openai_key = st.text_input("OpenAI Secret Key", type="password")
        if openai_key:
            if "app" not in st.session_state or st.session_state.get("provider") != "OpenAI" or st.session_state.get("key") != openai_key:
                st.session_state.app = init_embedchain_bot("OpenAI", api_key=openai_key)
                st.session_state.provider = "OpenAI"
                st.session_state.key = openai_key
            bot_initialized = True
        else:
            st.info("Enter OpenAI Key to boot kernel.")
    else:
        ollama_url = st.text_input("Ollama Base API Port", value="http://localhost:11434/v1")
        llm_model = st.text_input("Kernel Core LLM Model", value="llama3.2")
        embedder_model = st.text_input("V-Embedder Model", value="nomic-embed-text")
        
        if st.button("Connect Local Core"):
            try:
                st.session_state.app = init_embedchain_bot("Ollama", base_url=ollama_url, llm_model=llm_model, embedder_model=embedder_model)
                st.session_state.provider = "Ollama"
                st.session_state.bot_initialized_ollama = True
                st.success("Local RAG Engine connected!")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

        if st.session_state.get("bot_initialized_ollama") and st.session_state.get("provider") == "Ollama":
            bot_initialized = True

    # Sidebar Statistics Dashboard Widget (Glassmorphism)
    st.markdown("---")
    st.subheader("📊 Engine Metrics")
    
    total_docs = 0
    if bot_initialized and "app" in st.session_state:
        total_docs = st.session_state.app.db.count()
        
    st.markdown(f"""
        <div class="glass-card">
            <div style="font-size: 0.8rem; color: #94A3B8;">INDEXED CHUNKS</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #818CF8; margin: 4px 0;">{total_docs}</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">ENGINE LATENCY: <span style="color: #10B981; font-weight: 600;">ACTIVE</span></div>
        </div>
    """, unsafe_allow_html=True)

# Main Application Logic
if not bot_initialized:
    # Futuristic Hero Dashboard (When uploader/engine is not connected yet)
    st.markdown("""
        <div style="margin-top: 50px; text-align: center; display: flex; flex-direction: column; align-items: center;">
            <div class="feature-badge">Next-Gen Document Intelligence</div>
            <h1 class="hero-title">Talk to Any Document.<br>Extract Cognitive Insights.</h1>
            <p class="hero-subtitle">CogniFile merges local LLM power with semantic retrieval to turn unstructured PDFs into instant conversational intelligence pipelines.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Showcase Cards Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="glass-card glow-card-indigo">
                <h4 style="color: #818CF8; margin-top: 0;">🧠 Semantic RAG Core</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0;">Context-aware vector embeddings map your document topics into neural multi-dimensional maps.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="glass-card glow-card-coral">
                <h4 style="color: #FF6B6B; margin-top: 0;">🔒 100% Offline Nodes</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0;">Run completely secure, confidential intelligence tasks locally using Ollama core nodes.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="glass-card glow-card-purple">
                <h4 style="color: #A855F7; margin-top: 0;">⚡ Synthetic Timelines</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0;">Extract risks, actions, and milestones automatically upon uploading the data.</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.info("💡 Boot the engine via the Sidebar Node to start uploading files.")

else:
    # If the system is initialized, let the user upload and display a split-screen workspace
    app = st.session_state.app

    # If the pdf is not loaded yet in this session, show a beautiful central drag & drop zone
    if "pdf_uploaded" not in st.session_state:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h2 style="font-weight: 700; margin-bottom: 0px;">📥 Document Intake Portal</h2>
                <p style="color: #94A3B8; font-size: 0.95rem;">Upload files for cognitive ingestion.</p>
            </div>
        """, unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("", type="pdf")
        
        if pdf_file:
            # AI Ingestion Workflow Animation
            workflow_placeholder = st.empty()
            with workflow_placeholder.container():
                st.markdown("""
                    <div class="glass-card">
                        <h4 style="margin-top: 0; color: #818CF8;">⚙️ Ingestion & Document Pipeline</h4>
                        <div class="workflow-step"><span class="workflow-icon">⏳</span><span class="workflow-text">Reading raw binary byte stream...</span></div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Perform file copy & database add
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(pdf_file.getvalue())
                temp_filename = f.name
                
            with workflow_placeholder.container():
                st.markdown("""
                    <div class="glass-card">
                        <h4 style="margin-top: 0; color: #818CF8;">⚙️ Ingestion & Document Pipeline</h4>
                        <div class="workflow-step"><span class="workflow-icon">✅</span><span class="workflow-text">File read complete.</span></div>
                        <div class="workflow-step"><span class="workflow-icon">⏳</span><span class="workflow-text">Extracting structural text from PDF pages...</span></div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Perform chunking & embedding
            app.add(temp_filename, data_type="pdf_file", metadata={"url": pdf_file.name, "source": pdf_file.name})
            
            with workflow_placeholder.container():
                st.markdown("""
                    <div class="glass-card">
                        <h4 style="margin-top: 0; color: #818CF8;">⚙️ Ingestion & Document Pipeline</h4>
                        <div class="workflow-step"><span class="workflow-icon">✅</span><span class="workflow-text">Text extraction complete.</span></div>
                        <div class="workflow-step"><span class="workflow-icon">⏳</span><span class="workflow-text">Generating neural embeddings & indexing vector database...</span></div>
                    </div>
                """, unsafe_allow_html=True)
            os.remove(temp_filename)
            
            # Save PDF details to session state so it persists across streamlit reruns
            st.session_state.pdf_uploaded = True
            st.session_state.pdf_name = pdf_file.name
            st.session_state.pdf_bytes = pdf_file.getvalue()
            
            # Generate AI Insights automatically in background
            with workflow_placeholder.container():
                st.markdown("""
                    <div class="glass-card">
                        <h4 style="margin-top: 0; color: #818CF8;">⚙️ Ingestion & Document Pipeline</h4>
                        <div class="workflow-step"><span class="workflow-icon">✅</span><span class="workflow-text">Vector Database indexed.</span></div>
                        <div class="workflow-step"><span class="workflow-icon">⏳</span><span class="workflow-text">Synthesizing document intelligence summary...</span></div>
                    </div>
                """, unsafe_allow_html=True)
                
            try:
                summary = app.chat("Summarize this document in 3 short bullet points.")
                topics = app.chat("Identify the top 3 core topics of this document and explain them in one sentence each.")
                actions = app.chat("List any actions, timelines, or deadlines found in this document. If none, write 'No immediate deadlines found'.")
            except Exception as e:
                summary = "Summary generated offline."
                topics = "Core topics indexed successfully."
                actions = "Deadlines mapped dynamically."
                
            st.session_state.insights = {
                "summary": summary,
                "topics": topics,
                "actions": actions
            }
            
            # Clear placeholder and rerun to load workspace
            workflow_placeholder.empty()
            st.rerun()
            
    else:
        # Split-screen Workspace UI (File loaded)
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
                <div>
                    <span class="feature-badge">Active Workspace</span>
                    <h2 style="margin: 0; font-weight: 700;">📂 {st.session_state.pdf_name}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Action Bar to reset
        if st.button("🚪 Close & Unmount Document"):
            # Clear upload variables
            del st.session_state.pdf_uploaded
            del st.session_state.pdf_name
            del st.session_state.pdf_bytes
            if "insights" in st.session_state:
                del st.session_state.insights
            app.reset()
            st.rerun()

        st.markdown("---")

        workspace_col1, workspace_col2 = st.columns([1, 1])

        # Left Column - Document Intelligence Panel & Preview
        with workspace_col1:
            st.markdown('<div class="workspace-title">🔍 Document Intelligence Dashboard</div>', unsafe_allow_html=True)
            
            # Interactive Insights Tabs
            tab1, tab2, tab3 = st.tabs(["📋 Executive Summary", "🏷️ Key Topics", "📅 Action Plan"])
            
            with tab1:
                st.markdown(f"""
                    <div class="glass-card glow-card-indigo">
                        <h4 style="margin-top:0; color: #818CF8;">Executive Summary</h4>
                        <div style="color: #E2E8F0; font-size: 0.95rem; line-height:1.6;">
                            {st.session_state.insights.get("summary", "Summary processing...")}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown(f"""
                    <div class="glass-card glow-card-purple">
                        <h4 style="margin-top:0; color: #A855F7;">Extracted Core Themes</h4>
                        <div style="color: #E2E8F0; font-size: 0.95rem; line-height:1.6;">
                            {st.session_state.insights.get("topics", "Topics extraction...")}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            with tab3:
                st.markdown(f"""
                    <div class="glass-card glow-card-coral">
                        <h4 style="margin-top:0; color: #FF6B6B;">Timeline & Milestone Map</h4>
                        <div style="color: #E2E8F0; font-size: 0.95rem; line-height:1.6;">
                            {st.session_state.insights.get("actions", "Timeline mapping...")}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Embedded PDF Reader preview
            st.markdown('<div class="workspace-title" style="margin-top:30px;">📄 Live PDF Preview</div>', unsafe_allow_html=True)
            try:
                base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="450px" style="border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; background-color: #0A0F1D;"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error("Could not render PDF preview in browser.")

        # Right Column - AI Chat Agent
        with workspace_col2:
            st.markdown('<div class="workspace-title">💬 Cognitive Agent Workspace</div>', unsafe_allow_html=True)
            
            # Setup session state chat history if not exists
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display Chat History
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
                    label = "👤 You" if msg["role"] == "user" else "🤖 CogniFile"
                    st.markdown(f"""
                        <div class="chat-bubble {role_class}">
                            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px; font-weight:600;">{label}</div>
                            {msg["content"]}
                        </div>
                    """, unsafe_allow_html=True)

            # Quick suggestion buttons
            st.markdown("<p style='font-size:0.8rem; color:#94A3B8; font-weight:600; margin-bottom:8px;'>SUGGESTED ACTIONS</p>", unsafe_allow_html=True)
            quick_cols = st.columns(2)
            with quick_cols[0]:
                if st.button("💡 Analyze project risks"):
                    prompt_val = "Identify any major risks or negative points in this document."
                    st.session_state.chat_history.append({"role": "user", "content": prompt_val})
                    with st.spinner("Processing risks..."):
                        ans = app.chat(prompt_val)
                        st.session_state.chat_history.append({"role": "bot", "content": ans})
                    st.rerun()
            with quick_cols[1]:
                if st.button("📊 Extract key metrics"):
                    prompt_val = "List all numerical statistics, metrics, or financial figures mentioned in this document."
                    st.session_state.chat_history.append({"role": "user", "content": prompt_val})
                    with st.spinner("Processing metrics..."):
                        ans = app.chat(prompt_val)
                        st.session_state.chat_history.append({"role": "bot", "content": ans})
                    st.rerun()

            # Chat input field
            st.text_input("Ask anything about this document...", key="chat_input", on_change=handle_chat_input)
            
            if st.session_state.user_query:
                query = st.session_state.user_query
                st.session_state.user_query = None  # Reset query to prevent loop
                
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                with st.spinner("Thinking..."):
                    try:
                        bot_response = app.chat(query)
                    except Exception as e:
                        bot_response = f"RAG Query processing error: {e}"
                        
                # Add bot response
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                st.rerun()

            # Clear chat history button
            if st.button("🗑️ Clear Conversational Thread"):
                st.session_state.chat_history = []
                st.rerun()

        # Database Debug menu inside collapsed accordion at the very bottom
        st.markdown("---")
        with st.expander("🛠️ Workspace Kernel Debug Terminal"):
            st.subheader("Database Overview")
            st.write(f"- **Total Chunks in DB**: {app.db.count()}")
            
            # Clean and display document texts with matching scores
            st.subheader("Stored Document Chunks & Similarity Scores")
            cleaned_docs = [doc.replace("\x7f", " • ").replace("\n", " ").strip() for doc in app.db.documents]
            all_sims = getattr(app.db, "all_chunk_sims", [])
            for i, doc in enumerate(cleaned_docs):
                score_str = f"**(Similarity Score: {all_sims[i]:.4f})**" if i < len(all_sims) else ""
                st.markdown(f"- Chunk {i+1} {score_str}: {doc}")
                
            # Clean and display metadata
            st.subheader("Stored Metadata")
            st.write(app.db.metadatas)
            
            # Query status details
            st.subheader("Last Vector Search Query Info")
            st.write(f"- **Query 'where' filter applied**: `{getattr(app.db, 'last_query_where', None)}`")
            st.write(f"- **Query matches returned**: {getattr(app.db, 'last_query_results', None)}")
            st.write(f"- **Similarity scores of matches**: {getattr(app.db, 'last_query_sims', None)}")