"""
NexusMind – Document Ingestion
Supports: PDF, DOCX, TXT
Returns list of chunk dicts ready for embedding.
"""

import re
import html
from pathlib import Path
from typing import List, Dict

from backend.config import CHUNK_WORDS, CHUNK_OVERLAP


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    elif suffix == ".docx":
        return _extract_docx(file_path)
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(file_path: str) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        # fallback: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber")


def _extract_docx(file_path: str) -> str:
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # Remove non-printable chars
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    # Normalize whitespace (preserve paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_frac: float = CHUNK_OVERLAP,
) -> List[Dict]:
    if not text:
        return []

    words = text.split()
    n = len(words)

    if n <= chunk_words:
        return [{"chunk_index": 0, "chunk_text": text, "chunk_words": n}]

    overlap = max(1, int(chunk_words * overlap_frac))
    step    = chunk_words - overlap
    chunks  = []
    idx     = 0

    for start in range(0, n, step):
        chunk_words_list = words[start : start + chunk_words]
        if not chunk_words_list:
            break
        chunks.append({
            "chunk_index": idx,
            "chunk_text":  " ".join(chunk_words_list),
            "chunk_words": len(chunk_words_list),
        })
        idx += 1
        if start + chunk_words >= n:
            break

    return chunks


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest_document(file_path: str) -> List[Dict]:
    """
    Full ingestion pipeline:
    file → raw text → clean → chunk → list[Dict]
    """
    raw_text   = extract_text(file_path)
    clean      = clean_text(raw_text)
    chunks     = split_into_chunks(clean)
    return chunks
