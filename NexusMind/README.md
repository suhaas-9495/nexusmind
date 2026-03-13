# 🧠 NexusMind — Agentic RAG Application

> **Upload any PDF, DOCX, or TXT — then chat with it using a fully local, privacy-first AI stack.**  
> Powered by **Ollama LLM** · **Qdrant Vector Store** · **Agentic Planning & Tool Use**

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-format ingestion** | PDF (pdfplumber), DOCX, TXT |
| 🔢 **Local embeddings** | Ollama `nomic-embed-text` (768-dim) |
| 🗃️ **Vector search** | Qdrant cosine similarity |
| 🤖 **Agentic mode** | LLM plans → multi-step tool execution → synthesized answer |
| 💬 **RAG chat mode** | Direct retrieval-augmented Q&A with source citations |
| 🧠 **Conversation memory** | Multi-turn context window (last 10 turns) |
| 🔀 **Hybrid search** | BM25 + Semantic → Reciprocal Rank Fusion (RRF) |
| 🔒 **100% local** | No data sent to any cloud — fully private |
| 📊 **Streamlit UI** | Clean, professional web interface |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     NexusMind                           │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐  │
│  │  Upload  │───▶│ Ingest   │───▶│  Embed (Ollama)   │  │
│  │ PDF/DOCX │    │ & Chunk  │    │  nomic-embed-text │  │
│  └──────────┘    └──────────┘    └─────────┬─────────┘  │
│                                            │             │
│                                     ┌──────▼──────┐     │
│                                     │   Qdrant    │     │
│                                     │ Vector Store│     │
│                                     └──────┬──────┘     │
│                                            │             │
│  ┌─────────────────────┐          ┌────────▼────────┐   │
│  │   Agentic Mode      │          │  RAG Chat Mode  │   │
│  │ Plan → Execute →    │          │ Query → Retrieve│   │
│  │ Retrieve → Synthesize│         │ → LLM Answer    │   │
│  └─────────────────────┘          └─────────────────┘   │
│                                            │             │
│                                   ┌────────▼────────┐   │
│                                   │  Ollama LLM     │   │
│                                   │  (llama3, etc.) │   │
│                                   └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3
ollama pull nomic-embed-text

# Start Qdrant via Docker
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install Python dependencies

```bash
git clone https://github.com/yourname/nexusmind
cd nexusmind

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 4. Run the application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📖 Usage

### Step 1 — Upload a document
Click **Browse files** in the sidebar → select a PDF, DOCX, or TXT file → click **Index Documents**.

### Step 2 — Choose a mode

| Mode | When to use |
|---|---|
| **RAG Chat** | Fast, direct Q&A — retrieves chunks and answers immediately |
| **Agentic Mode** | Complex questions — LLM plans multi-step retrieval and reasoning |

### Step 3 — Ask questions
Type your question in the chat box and press **Send**. The answer includes:
- ✅ The answer text (with inline `[Source N]` citations)
- 📖 Expandable source cards showing which chunks were used
- 🔍 (Agentic) Expandable agent reasoning steps

---

## 📁 Project Structure

```
nexusmind/
├── app.py                    # Streamlit web application
├── requirements.txt
├── .env.example
│
├── backend/
│   ├── config.py             # Central configuration
│   ├── ingestion.py          # PDF/DOCX/TXT extraction + chunking
│   ├── embedding.py          # Ollama embedding wrapper
│   ├── indexing.py           # Qdrant upsert + collection management
│   ├── retrieval.py          # Qdrant similarity search
│   └── llm_answer.py         # Ollama chat (blocking + streaming)
│
├── agent/
│   ├── runner.py             # Main agentic orchestration loop
│   ├── planner.py            # LLM-based step planner
│   ├── executor.py           # Tool dispatcher
│   ├── memory.py             # Conversation + step memory
│   └── state.py              # Agent state dataclass
│
├── tools/
│   ├── rag_tool.py           # Retrieve from vector store
│   ├── calculator.py         # Safe math evaluator
│   ├── summarizer.py         # Summarize step results
│   ├── analyzer.py           # Analyze retrieved data
│   └── synthesizer.py        # Combine findings into answer
│
├── logs/
│   └── agent_logs.jsonl      # Structured agent execution logs
│
└── uploads/                  # Temporary document storage
```

---

## ⚙️ Configuration

All settings are in `.env` (or environment variables):

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `TEMPERATURE` | `0.1` | LLM temperature |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `COLLECTION_NAME` | `nexusmind_docs` | Qdrant collection |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_WORDS` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `0.15` | Chunk overlap fraction |
| `AGENT_MAX_STEPS` | `6` | Max agent steps per run |

---

## 🧩 Tech Stack

| Component | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) (`llama3`, `mistral`, etc.) |
| Embeddings | Ollama `nomic-embed-text` |
| Vector Store | [Qdrant](https://qdrant.tech) |
| UI | [Streamlit](https://streamlit.io) |
| PDF parsing | pdfplumber / pypdf |
| DOCX parsing | python-docx |

---

## 🔄 Supported Ollama Models

| Model | Size | Best for |
|---|---|---|
| `llama3` | 8B | General Q&A (default) |
| `llama3:70b` | 70B | Highest accuracy |
| `mistral` | 7B | Fast inference |
| `gemma2` | 9B | Multi-lingual |
| `phi3` | 3.8B | Low-resource machines |

Change the model from the **⚙️ Model Settings** panel in the sidebar — no restart needed.

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ — NexusMind merges production-grade RAG pipelines with an agentic AI framework, all running 100% locally.*
