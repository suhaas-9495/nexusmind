"""
NexusMind v2 — Central Configuration
All values readable from environment / .env file.
Backward-compatible module-level names preserved.
"""

import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    OLLAMA_BASE_URL:   str   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
    LLM_MODEL:         str   = os.getenv("LLM_MODEL",         "llama3")
    EMBED_MODEL:       str   = os.getenv("EMBED_MODEL",        "nomic-embed-text")
    TEMPERATURE:       float = float(os.getenv("TEMPERATURE",  "0.1"))
    QDRANT_URL:        str   = os.getenv("QDRANT_URL",         "http://localhost:6333")
    COLLECTION_NAME:   str   = os.getenv("COLLECTION_NAME",    "nexusmind_docs")
    VECTOR_SIZE:       int   = int(os.getenv("VECTOR_SIZE",    "768"))
    TOP_K:             int   = int(os.getenv("TOP_K",          "5"))
    CHUNK_WORDS:       int   = int(os.getenv("CHUNK_WORDS",    "400"))
    CHUNK_OVERLAP:     float = float(os.getenv("CHUNK_OVERLAP","0.15"))
    AGENT_MAX_STEPS:   int   = int(os.getenv("AGENT_MAX_STEPS",   "6"))
    AGENT_MAX_RETRIES: int   = int(os.getenv("AGENT_MAX_RETRIES", "2"))
    AGENT_TIMEOUT_SEC: int   = int(os.getenv("AGENT_TIMEOUT_SEC", "120"))
    JWT_SECRET_KEY:    str   = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    JWT_EXPIRE_SECS:   int   = int(os.getenv("JWT_EXPIRE_SECS", "86400"))

settings = Settings()

# Backward-compatible module-level names
OLLAMA_BASE_URL   = settings.OLLAMA_BASE_URL
LLM_MODEL         = settings.LLM_MODEL
EMBED_MODEL       = settings.EMBED_MODEL
TEMPERATURE       = settings.TEMPERATURE
QDRANT_URL        = settings.QDRANT_URL
COLLECTION_NAME   = settings.COLLECTION_NAME
VECTOR_SIZE       = settings.VECTOR_SIZE
TOP_K             = settings.TOP_K
CHUNK_WORDS       = settings.CHUNK_WORDS
CHUNK_OVERLAP     = settings.CHUNK_OVERLAP
AGENT_MAX_STEPS   = settings.AGENT_MAX_STEPS
AGENT_MAX_RETRIES = settings.AGENT_MAX_RETRIES
AGENT_TIMEOUT_SEC = settings.AGENT_TIMEOUT_SEC
