# 📄 RAG Document QnA

A **Retrieval-Augmented Generation (RAG)** application that lets you query the content of PDF documents using natural language. Powered by **LangChain**, **Groq LLM (Llama-3)**, and **HuggingFace Embeddings**, it provides accurate, context-aware answers directly from your documents.

---

## ✨ Features

- 📚 Load and process PDF documents for intelligent Q&A
- 🔍 Semantic search using FAISS or ChromaDB vector stores
- 🤖 LLM-powered answers via **Groq** (Llama-3)
- 🧠 HuggingFace `all-MiniLM-L6-v2` embeddings for accurate retrieval
- 💬 Two app versions:
  - **v1** — Directory-based PDF loading with FAISS
  - **v2** — Upload-your-own PDFs with conversation history (ChromaDB)
- 🌐 Clean and interactive **Streamlit** UI

---

## 🗂️ Project Structure

```
rag_document_qna/
│
├── app_v1.py              # Version 1: Loads PDFs from a local directory (FAISS)
├── app_v2.py              # Version 2: Upload PDFs + conversational memory (Chroma)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── .gitignore             # Git ignore rules
└── research_papers/       # Sample PDFs for app_v1
    ├── Attention.pdf      # "Attention Is All You Need" paper
    └── LLM.pdf            # LLM research paper
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag_document_qna
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root (if not already present) and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> 🔑 Get your **Groq API key** at [https://console.groq.com](https://console.groq.com)
> 🔑 Get your **HuggingFace token** at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🖥️ Running the App

### Version 1 — Directory-Based PDF Q&A (FAISS)

Loads PDFs automatically from the `research_papers/` folder.

```bash
streamlit run app_v1.py
```

**How it works:**
1. Click **"Document Embeddings"** to process and embed the PDFs.
2. Type your question in the input box.
3. The app retrieves relevant chunks and returns an LLM-powered answer.
4. Expand **"Document similarity search"** to view the source chunks used.

---

### Version 2 — Upload PDFs + Conversational Memory (ChromaDB)

Allows you to upload your own PDFs and chat with them in a multi-turn conversation.

```bash
streamlit run app_v2.py
```

**How it works:**
1. Enter your **Groq API key** in the sidebar input.
2. (Optional) Set a **Session ID** to manage separate conversation histories.
3. Upload one or more **PDF files**.
4. Ask questions — the app remembers your conversation history for context-aware follow-ups.

---

## 🛠️ Tech Stack

| Component            | Technology                              |
|---------------------|-----------------------------------------|
| UI Framework         | [Streamlit](https://streamlit.io)       |
| LLM                  | [Groq](https://groq.com) (Llama-3)     |
| Embeddings           | HuggingFace `all-MiniLM-L6-v2`         |
| Vector Store (v1)    | FAISS                                   |
| Vector Store (v2)    | ChromaDB                                |
| Document Loader      | LangChain `PyPDFDirectoryLoader` / `PyPDFLoader` |
| Text Splitter        | `RecursiveCharacterTextSplitter`        |
| Orchestration        | [LangChain](https://www.langchain.com/) |

---

## 📦 Dependencies

```
langchain
langchain_core
langchain_community
langchain_groq
langchain_huggingface
langchain_chroma
langchain_openai
streamlit
python-dotenv
pypdf
sentence_transformers
faiss-cpu
chroma
```

Install all with:

```bash 
pip install -r requirements.txt
```

---

## ⚙️ App Comparison

| Feature                   | app_v1.py          | app_v2.py                  |
|--------------------------|--------------------|-----------------------------|
| PDF Source               | Local directory    | User file upload            |
| Vector Store             | FAISS              | ChromaDB                    |
| Conversation Memory      | ❌ None            | ✅ Session-based history    |
| API Key Input            | `.env` only        | UI input + `.env`           |
| Multi-turn Chat          | ❌                 | ✅                          |

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/) — LLM application framework
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [HuggingFace](https://huggingface.co/) — Open-source embeddings
- [Streamlit](https://streamlit.io/) — Rapid web UI for ML apps
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Sample research paper
