"""
NexusMind v2 — /upload-doc endpoint
Async document ingestion pipeline.
"""

import asyncio
import time
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List

from auth.jwt_handler import get_current_user
from backend.ingestion import ingest_document
from backend.embedding import embed_texts
from backend.indexing import ensure_collection, index_chunks
from observability.logger import get_request_logger
from observability.metrics import record_upload

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {
    "application/pdf", "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    status: str
    elapsed_sec: float
    user_id: str


async def _process_async(file_path: Path, filename: str, user_id: str) -> UploadResponse:
    """Run ingestion pipeline in a thread pool (CPU-bound work)."""
    start = time.perf_counter()
    loop = asyncio.get_event_loop()

    # Ingest in thread pool to avoid blocking event loop
    chunks = await loop.run_in_executor(None, ingest_document, str(file_path), user_id)

    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    texts = [c["chunk_text"] for c in chunks]
    embeddings = await loop.run_in_executor(None, embed_texts, texts)

    ensure_collection()
    meta = [
        {
            "id":           i,
            "chunk_text":   c["chunk_text"],
            "source":       filename,
            "chunk_index":  c["chunk_index"],
            "user_id":      user_id,
        }
        for i, c in enumerate(chunks)
    ]
    await loop.run_in_executor(None, index_chunks, embeddings, meta)

    elapsed = round(time.perf_counter() - start, 2)
    record_upload(filename=filename, chunks=len(chunks), elapsed=elapsed)

    return UploadResponse(
        filename=filename,
        chunks=len(chunks),
        status="indexed",
        elapsed_sec=elapsed,
        user_id=user_id,
    )


@router.post("/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload and index a document (PDF, TXT, DOCX).
    Requires JWT authentication.
    """
    # Validate content type
    if file.content_type not in ALLOWED_TYPES and not file.filename.endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF, TXT, or DOCX."
        )

    # Read and size-check
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    # Save to uploads/
    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / f"{current_user['sub']}_{safe_name}"
    dest.write_bytes(content)

    try:
        result = await _process_async(dest, safe_name, current_user["sub"])
        get_request_logger().info(
            f"User={current_user['sub']} uploaded '{safe_name}' → {result.chunks} chunks"
        )
        return result
    except Exception as e:
        logger.error(f"Upload failed for '{safe_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
