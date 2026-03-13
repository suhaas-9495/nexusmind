"""
NexusMind – Central Configuration
All settings read from environment variables (with sensible defaults).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL",        "llama3")
EMBED_MODEL       = os.getenv("EMBED_MODEL",      "nomic-embed-text")
TEMPERATURE       = float(os.getenv("TEMPERATURE", "0.1"))

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL",  "http://localhost:6333")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "nexusmind_docs")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", "768"))   # nomic-embed-text = 768

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K             = int(os.getenv("TOP_K", "5"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_WORDS       = int(os.getenv("CHUNK_WORDS",   "400"))
CHUNK_OVERLAP     = float(os.getenv("CHUNK_OVERLAP", "0.15"))

# ── Agent ─────────────────────────────────────────────────────────────────────
AGENT_MAX_STEPS   = int(os.getenv("AGENT_MAX_STEPS",   "6"))
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))
AGENT_TIMEOUT_SEC = int(os.getenv("AGENT_TIMEOUT_SEC", "120"))
