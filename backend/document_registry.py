from pathlib import Path
from typing import Optional

from backend.ingestion import generate_hash


class DocumentRegistry:
    """
    Temporary in-memory document registry.
    Later this class will use Qdrant/PostgreSQL instead of Python memory.
    """

    def __init__(self):
        self._documents = {}

    def register_document(
        self,
        file_path: str,
        user_id: Optional[str] = None,
    ) -> dict:

        path = Path(file_path)

        document_hash = generate_hash(path)

        key = (
            user_id,
            path.name,
        )

        if key not in self._documents:

            self._documents[key] = [
                {
                    "version": 1,
                    "hash": document_hash,
                    "active": True,
                }
            ]

            return {
                "duplicate": False,
                "version": 1,
                "document_hash": document_hash,
            }

        latest = self._documents[key][-1]

        if latest["hash"] == document_hash:

            return {
                "duplicate": True,
                "version": latest["version"],
                "document_hash": document_hash,
            }

        latest["active"] = False

        new_version = latest["version"] + 1

        self._documents[key].append(
            {
                "version": new_version,
                "hash": document_hash,
                "active": True,
            }
        )

        return {
            "duplicate": False,
            "version": new_version,
            "document_hash": document_hash,
        }


registry = DocumentRegistry()