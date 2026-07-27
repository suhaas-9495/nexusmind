from collections import defaultdict
from typing import Dict, List


class ReciprocalRankFusion:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: List[List[Dict]],
        weights: List[float] | None = None,
        id_key: str = "chunk_index",
        source_key: str = "source",
    ) -> List[Dict]:

        if not ranked_lists:
            return []

        if weights is None:
            weights = [1.0] * len(ranked_lists)

        scores = defaultdict(float)
        chunks = {}

        for ranked_list, weight in zip(ranked_lists, weights):

            for rank, chunk in enumerate(ranked_list, start=1):

                uid = (
                    chunk.get(source_key),
                    chunk.get(id_key),
                )

                scores[uid] += weight * (
                    1.0 / (self.k + rank)
                )

                if uid not in chunks:
                    chunks[uid] = chunk

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for uid, score in ranked:

            item = dict(chunks[uid])

            item["rrf_score"] = round(score, 6)

            results.append(item)

        return results