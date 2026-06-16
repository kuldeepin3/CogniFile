# 📄 CogniFile: AI-Powered Document Intelligence Platform

**CogniFile** is a premium, next-generation AI operating system for documents. It runs a secure, semantic RAG (Retrieval-Augmented Generation) pipeline entirely locally and offline, allowing you to upload, analyze, and chat with your PDFs without needing paid API keys or exposing sensitive data.

---

## ✨ Features

* **🧠 Semantic RAG Core**: Utilizes neural vector embeddings to map document chunks into multi-dimensional databases for context-aware question answering.
* **🔒 100% Offline Mode**: Supported via **Ollama** (`llama3.2` + `nomic-embed-text`) routing all embeddings and LLM prompts locally—no data leaves your machine.
* **💾 Custom Pure-Python Vector Store (`InMemoryDB`)**: Built using pure Python and NumPy to bypass compilation dependencies (like `chroma-hnswlib`), making it cross-platform and extremely lightweight.
* **⚡ Split-Screen Workspace**: Side-by-side cockpit interface featuring:
  * **Left Panel**: Interactive Live PDF preview alongside auto-synthesized AI Insight Cards (**Executive Summary**, **Core Themes**, and **Milestone Timelines**).
  * **Right Panel**: Clean, responsive conversational agent thread with quick-action suggestion prompts.
* **⚙️ Visual Ingestion Pipeline**: Delightful step-by-step progress tracking (*Reading raw file -> Extracting text -> Building neural embeddings*).
* **📊 Live Metrics Widget**: Sidebar metrics tracker showing total indexed chunks, node status, and engine latency.

---

## 🛠️ Tech Stack

* **Frontend & Dashboard**: Streamlit (with customized HTML/CSS glassmorphic styles)
* **Orchestration**: Embedchain & LangChain
* **Vector Mathematics**: NumPy
* **PDF Parsing**: PyPDF
* **Supported Models**: OpenAI (Cloud) or Ollama (Local)

---

## 🚀 Getting Started

### 1. Ingest Dependencies
Ensure you have the required packages installed:
```bash
pip install streamlit embedchain streamlit-chat numpy pypdf posthog openai==0.28
```

### 2. Configure Local LLM Node (Ollama)
If running locally (free & secure):
1. Download and start [Ollama](https://ollama.com/).
2. Pull the LLM and Embedding models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Run `ollama serve`.

### 3. Run the Platform
Navigate to the directory and boot the Streamlit server:
```bash
python -m streamlit run chat_pdf.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ⚙️ How to Connect

1. In the left sidebar under **API Engine Node**, choose **Local Ollama Engine**.
2. Click **Connect Local Core**.
3. Upload your PDF file.
4. Synthesize your insights and start asking questions!
