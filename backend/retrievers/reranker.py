from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from backend.config import CROSS_ENCODER_MODEL


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name or CROSS_ENCODER_MODEL
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """
        Lazily load the model only when first used.
        """
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int,
    ) -> List[Dict]:

        if not documents:
            return []

        valid_docs = [
            doc
            for doc in documents
            if doc.get("chunk_text")
        ]

        if not valid_docs:
            return []

        pairs = [
            (query, doc["chunk_text"])
            for doc in valid_docs
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = []

        for doc, score in zip(valid_docs, scores):
            item = dict(doc)
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]