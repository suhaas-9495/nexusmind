"""
NexusMind v2 — Document Ingestion
Supports: PDF, DOCX, TXT. Tags chunks with user_id for access control.
"""
import re, html
from pathlib import Path
from typing import List, Dict, Optional
from backend.config import CHUNK_WORDS, CHUNK_OVERLAP

def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf": return _extract_pdf(file_path)
    elif suffix == ".docx": return _extract_docx(file_path)
    elif suffix in (".txt", ".md"): return path.read_text(encoding="utf-8", errors="ignore")
    else: raise ValueError(f"Unsupported: {suffix}")

def _extract_pdf(file_path):
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(file_path).pages)

def _extract_docx(file_path):
    from docx import Document
    return "\n\n".join(p.text for p in Document(file_path).paragraphs if p.text.strip())

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def split_into_chunks(text: str, chunk_words: int = CHUNK_WORDS, overlap_frac: float = CHUNK_OVERLAP) -> List[Dict]:
    if not text: return []
    words = text.split(); n = len(words)
    if n <= chunk_words: return [{"chunk_index": 0, "chunk_text": text, "chunk_words": n}]
    overlap = max(1, int(chunk_words * overlap_frac)); step = chunk_words - overlap
    chunks = []; idx = 0
    for start in range(0, n, step):
        w = words[start:start + chunk_words]
        if not w: break
        chunks.append({"chunk_index": idx, "chunk_text": " ".join(w), "chunk_words": len(w)})
        idx += 1
        if start + chunk_words >= n: break
    return chunks

def ingest_document(file_path: str, user_id: Optional[str] = None) -> List[Dict]:
    """file → clean text → chunks, optionally tagged with user_id."""
    chunks = split_into_chunks(clean_text(extract_text(file_path)))
    if user_id:
        for c in chunks: c["user_id"] = user_id
    return chunks
