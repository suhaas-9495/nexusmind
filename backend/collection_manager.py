from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
)
class CollectionManager:
    """
    Handles creation and validation of Qdrant collections.
    """
    def __init__(self,client: QdrantClient):
        self.client = client
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        collections = self.client.get_collections()
        existing = {
            c.name
            for c in collections.collections
        }
        if collection_name in existing:
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
    def delete_collection(self,collection_name: str) -> None:
        self.client.delete_collection(
            collection_name
        )
    def collection_exists(self,collection_name: str) -> bool:
        collections = self.client.get_collections()
        return any(
            c.name == collection_name
            for c in collections.collections
        )