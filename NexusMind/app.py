"""
NexusMind — Agentic RAG Application
Upload any PDF/TXT/DOCX and chat with it using a local Ollama LLM.
"""

import streamlit as st
import time
import json
import os
from pathlib import Path

# ── backend imports ──────────────────────────────────────────────────────────
from backend.ingestion        import ingest_document
from backend.embedding        import embed_texts
from backend.indexing         import ensure_collection, index_chunks
from backend.hybrid_retrieval import hybrid_retrieve
from backend.bm25_index       import load_bm25_index
from backend.llm_answer       import generate_answer, generate_answer_stream
from agent.runner import run_agent
from agent.memory import ConversationMemory

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusMind · Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }

    .chat-msg-user {
        background: #EEF2FF;
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .chat-msg-ai {
        background: #F0FDF4;
        border-left: 4px solid #22c55e;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .source-card {
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .agent-step {
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .status-ok   { color: #16a34a; font-weight: 600; }
    .status-warn { color: #d97706; font-weight: 600; }
    .status-err  { color: #dc2626; font-weight: 600; }

    .metric-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history":     [],
        "indexed_docs":     [],
        "memory":           ConversationMemory(max_turns=10),
        "collection_ready": False,
        "mode":             "RAG Chat",
        "use_hybrid":       True,
        "last_fusion_info": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Try loading persisted BM25 index (survives page reloads)
load_bm25_index()

# ── helpers ───────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> Path:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def process_document(file_path: Path) -> bool:
    """Ingest → chunk → embed → index. Returns True on success."""
    try:
        with st.spinner(f"📄 Reading & chunking `{file_path.name}` …"):
            chunks = ingest_document(str(file_path))

        if not chunks:
            st.error("No text extracted from document.")
            return False

        st.info(f"✂️  {len(chunks)} chunks created.")

        with st.spinner("🔢 Embedding chunks with Ollama …"):
            texts = [c["chunk_text"] for c in chunks]
            embeddings = embed_texts(texts)

        with st.spinner("💾 Indexing into Qdrant …"):
            ensure_collection()
            meta = [
                {
                    "id":         i,
                    "chunk_text": c["chunk_text"],
                    "source":     file_path.name,
                    "chunk_index": c["chunk_index"],
                }
                for i, c in enumerate(chunks)
            ]
            index_chunks(embeddings, meta)

        st.session_state.indexed_docs.append(file_path.name)
        st.session_state.collection_ready = True
        return True

    except Exception as e:
        st.error(f"Error during processing: {e}")
        return False


# ═══════════════════════════════ SIDEBAR ══════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NexusMind")
    st.caption("Agentic RAG · Powered by Ollama")

    st.divider()

    # ── Mode toggle ──
    st.session_state.mode = st.radio(
        "**Interaction Mode**",
        ["RAG Chat", "Agentic Mode"],
        help="RAG Chat: direct retrieval-augmented answers.\nAgentic Mode: multi-step planning + tool use.",
    )

    st.divider()

    # ── Model settings ──
    with st.expander("⚙️ Model Settings", expanded=False):
        llm_model   = st.text_input("LLM Model",   value="llama3")
        embed_model = st.text_input("Embed Model", value="nomic-embed-text")
        top_k       = st.slider("Top-K Chunks", 1, 10, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        st.session_state.use_hybrid = st.toggle(
            "🔀 Hybrid Search (BM25 + Semantic)",
            value=st.session_state.use_hybrid,
            help="Combines keyword (BM25) and semantic search via Reciprocal Rank Fusion for better accuracy.",
        )
        # Pass overrides to env so modules pick them up
        os.environ["LLM_MODEL"]   = llm_model
        os.environ["EMBED_MODEL"] = embed_model
        os.environ["TOP_K"]       = str(top_k)
        os.environ["TEMPERATURE"] = str(temperature)

    st.divider()

    # ── Document upload ──
    st.markdown("### 📂 Upload Documents")
    uploaded = st.file_uploader(
        "PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("🚀 Index Documents", type="primary", use_container_width=True):
            for f in uploaded:
                path = save_upload(f)
                ok = process_document(path)
                if ok:
                    st.success(f"✅ `{f.name}` indexed!")

    if st.session_state.indexed_docs:
        st.divider()
        st.markdown("**📚 Indexed Documents**")
        for d in st.session_state.indexed_docs:
            st.markdown(f"- `{d}`")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationMemory(max_turns=10)
        st.rerun()


# ═══════════════════════════════ MAIN AREA ════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧠 NexusMind — Agentic RAG</h1>
  <p>Upload documents · Ask questions · Get cited, intelligent answers</p>
</div>
""", unsafe_allow_html=True)

# ── metrics bar ──
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Documents", len(st.session_state.indexed_docs))
with col2:
    st.metric("Mode", st.session_state.mode.split()[0])
with col3:
    st.metric("LLM", os.environ.get("LLM_MODEL", "llama3"))
with col4:
    search_label = "Hybrid" if st.session_state.use_hybrid else "Semantic"
    st.metric("Search", search_label)
with col5:
    st.metric("Turns", len(st.session_state.chat_history))

st.divider()

# ── chat history display ──
chat_container = st.container()
with chat_container:
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">👤 <b>You:</b> {turn["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-ai">🧠 <b>NexusMind:</b></div>', unsafe_allow_html=True)
            st.markdown(turn["content"])

            # Agent steps
            if turn.get("agent_steps"):
                with st.expander("🔍 Agent Reasoning Steps"):
                    for step in turn["agent_steps"]:
                        st.markdown(f'<div class="agent-step">▶ {step}</div>', unsafe_allow_html=True)

            # Sources
            if turn.get("sources"):
                fusion_info = turn.get("fusion_info", {})
                label = f"📖 Sources ({len(turn['sources'])})"
                if fusion_info.get("fusion_method"):
                    label += f" · {fusion_info['fusion_method']}"
                with st.expander(label):
                    if fusion_info:
                        fc1, fc2, fc3 = st.columns(3)
                        fc1.metric("Semantic Hits", fusion_info.get("semantic_hits", "–"))
                        fc2.metric("BM25 Hits",    fusion_info.get("bm25_hits", "–"))
                        fc3.metric("Final Top-K",  len(turn["sources"]))
                        st.divider()
                    for src in turn["sources"]:
                        rrf_s  = src.get("rrf_score",  None)
                        sem_s  = src.get("score",       0)
                        bm25_s = src.get("bm25_score",  None)
                        score_parts = []
                        if rrf_s  is not None: score_parts.append(f"RRF={rrf_s:.4f}")
                        if sem_s:              score_parts.append(f"Sem={sem_s:.3f}")
                        if bm25_s is not None: score_parts.append(f"BM25={bm25_s:.2f}")
                        score_str = " · ".join(score_parts) or "–"
                        st.markdown(
                            f'<div class="source-card">'
                            f'📄 <b>{src.get("source","unknown")}</b> · '
                            f'Chunk #{src.get("chunk_index","?")} · '
                            f'{score_str}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src.get("chunk_text", "")[:250] + "…")

# ── input area ────────────────────────────────────────────────────────────────
st.divider()

if not st.session_state.collection_ready:
    st.warning("⬅️ Upload and index at least one document to start chatting.")
else:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask anything about your documents …",
            placeholder="e.g. 'Summarize the key findings' or 'What are the main risks?'",
            height=80,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

    if submitted and user_input.strip():
        query = user_input.strip()

        # Save user turn
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.memory.add("user", query)

        with st.spinner("🤔 Thinking …"):
            try:
                if st.session_state.mode == "Agentic Mode":
                    # ── Agentic path ────────────────────────────────────────
                    result = run_agent(query, st.session_state.memory)
                    answer      = result["final_answer"]
                    sources     = result.get("sources", [])
                    agent_steps = result.get("steps", [])
                else:
                    # ── RAG Chat path (Hybrid Retrieval) ─────────────────────
                    query_vec    = embed_texts([query])[0]
                    hybrid_res   = hybrid_retrieve(
                        query, query_vec,
                        top_k=int(os.environ.get("TOP_K", 5)),
                        use_bm25=st.session_state.use_hybrid,
                    )
                    chunks       = hybrid_res["results"]
                    fusion_info  = {
                        "fusion_method":  hybrid_res["fusion_method"],
                        "semantic_hits":  hybrid_res["semantic_hits"],
                        "bm25_hits":      hybrid_res["bm25_hits"],
                    }
                    answer       = generate_answer(query, chunks, st.session_state.memory.get_context())
                    sources      = chunks
                    agent_steps  = []

                st.session_state.memory.add("assistant", answer)
                st.session_state.chat_history.append({
                    "role":        "assistant",
                    "content":     answer,
                    "sources":     sources,
                    "agent_steps": agent_steps,
                    "fusion_info": fusion_info if st.session_state.mode == "RAG Chat" else {},
                })

            except Exception as e:
                err_msg = f"❌ Error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": [], "agent_steps": [], "fusion_info": {}})

        st.rerun()
