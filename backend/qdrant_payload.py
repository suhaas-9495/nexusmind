from uuid import uuid4
from typing import Dict, List

from qdrant_client.models import PointStruct

def build_qdrant_point(
    embedding: List[float],
    chunk: Dict,
) -> PointStruct:
    """
    like converting the chunk into Qdrant PointStruct.
    """
    return PointStruct(
        id=str(uuid4()),
        vector=embedding,
        payload={
            "document_id": chunk["document_id"],
            "document_hash": chunk["document_hash"],

            "file_name": chunk["file_name"],
            "file_extension": chunk["file_extension"],

            "uploaded_by": chunk["uploaded_by"],
            "uploaded_at": chunk["uploaded_at"],

            "version": chunk["version"],

            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],

            "chunk_words": chunk["chunk_words"],

            "start_word": chunk["start_word"],
            "end_word": chunk["end_word"],
            "tenant_id": chunk["tenant_id"],

            "tenant_name": chunk["tenant_name"],

            "owner_id": chunk["owner_id"],

            "token_estimate": chunk["token_estimate"],

            "embedding_model": chunk["embedding_model"],

            "is_active": chunk["is_active"],
        },
    )
    