from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


class BatchIngestion:

    """
    Upload vectors to Qdrant in configurable batches.
    """

    def __init__(self,client: QdrantClient,collection_name: str,batch_size: int = 100):
        self.client = client

        self.collection_name = collection_name

        self.batch_size = batch_size

    def upload(self,points: List[PointStruct]) -> None:
        if not points:
            return
        for start in range(
            0,
            len(points),
            self.batch_size,
        ):

            batch = points[
                start:start + self.batch_size
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )