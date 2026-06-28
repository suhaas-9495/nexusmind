"""
 Document Ingestion just support the necessary one nothing much in this code window !!!
Supports: PDF, DOCX, TXT.
"""
import re, html
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from uuid import uuid4
from datetime import datetime
from backend.config import CHUNK_WORDS, CHUNK_OVERLAP


MAX_FILE_SIZE = 50 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}



def validate_exists(path : Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    
def validate_size(path: Path):
    size = path.stat().st_size
    
    if size == 0:
        raise ValueError("Document is empty")
    if size > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB.")

def validate_extension(path: Path):
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: {path.suffix}"
        )


def generate_hash(path: Path):

    sha = hashlib.sha256()

    with open(path, "rb") as file:

        while chunk := file.read(8192):
            sha.update(chunk)

    return sha.hexdigest()


def build_document_metadata(
    file_path: str,
    document_hash: str,
    user_id: Optional[str] = None,
) -> Dict:

    path = Path(file_path)

    return {
        "document_id": str(uuid4()),
        "document_hash": document_hash,

        "file_name": path.name,
        "file_extension": path.suffix.lower(),
        "file_size": path.stat().st_size,

        "uploaded_at": datetime.now().astimezone().isoformat(),
        "uploaded_by": user_id,

        "version": 1,
    }

      
    
def extract_text(file_path: str) -> str:

    path = Path(file_path)

    validate_exists(path)
    validate_size(path)
    validate_extension(path)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(file_path)

    elif suffix == ".docx":
        text = _extract_docx(file_path)

    elif suffix in (".txt", ".md"):
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    else:
        raise ValueError(f"Unsupported: {suffix}")

    if len(text.strip()) < 100:
        raise ValueError(
            "Document contains too little text."
        )

    return text

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
    text = clean_text(extract_text(file_path))
    document_hash = generate_hash(Path(file_path))
    metadata["checksum"] = document_hash
    metadata = build_document_metadata(
        file_path=file_path,
        document_hash=document_hash,
        user_id=user_id,
    )

    chunks = split_into_chunks(text)

    for chunk in chunks:

        chunk.update(metadata)

    return chunks
