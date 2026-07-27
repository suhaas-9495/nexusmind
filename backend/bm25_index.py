"""
NexusMind – Production BM25 Index
"""

import math
import pickle
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import heapq


class BM25:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.corpus_size: int = 0
        self.avgdl: float = 0.0

        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        self._corpus_texts: List[str] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:

        if not text:
            return []

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        return [
            token
            for token in text.split()
            if len(token) > 1
        ]

    def fit(self, corpus: List[str]):

        self._corpus_texts = corpus
        self.corpus_size = len(corpus)

        tokenized = [
            self.tokenize(doc)
            for doc in corpus
        ]

        self.doc_len = [
            len(tokens)
            for tokens in tokenized
        ]

        self.avgdl = (
            sum(self.doc_len) / self.corpus_size
            if self.corpus_size
            else 1.0
        )

        self.doc_freqs = []

        document_frequency: Dict[str, int] = {}

        for tokens in tokenized:

            frequencies: Dict[str, int] = {}

            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            self.doc_freqs.append(frequencies)

            for token in frequencies:
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        self.idf.clear()

        for term, df in document_frequency.items():

            self.idf[term] = math.log(
                ((self.corpus_size - df + 0.5) / (df + 0.5)) + 1
            )

    def score(self, query: str, doc_index: int) -> float:

        query_terms = set(self.tokenize(query))

        if not query_terms:
            return 0.0

        frequencies = self.doc_freqs[doc_index]

        document_length = self.doc_len[doc_index]

        score = 0.0

        for term in query_terms:

            tf = frequencies.get(term)

            if tf is None:
                continue

            idf = self.idf.get(term, 0.0)

            numerator = tf * (self.k1 + 1)

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b * document_length / self.avgdl
                )
            )

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:

        if self.corpus_size == 0:
            return []

        scores = (
            (index, self.score(query, index))
            for index in range(self.corpus_size)
        )

        top = heapq.nlargest(
            top_k,
            scores,
            key=lambda x: x[1],
        )

        return [
            (index, score)
            for index, score in top
            if score > 0
        ]


_bm25_index: Optional[BM25] = None
_bm25_chunks: Optional[List[Dict]] = None

CACHE_PATH = Path("logs/bm25_cache.pkl")


def build_bm25_index(chunks: List[Dict]):

    global _bm25_index
    global _bm25_chunks

    texts = [
        chunk.get("chunk_text", "")
        for chunk in chunks
    ]

    _bm25_index = BM25()
    _bm25_index.fit(texts)

    _bm25_chunks = list(chunks)

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp:

        pickle.dump(
            (_bm25_index, _bm25_chunks),
            temp,
        )

        temp_path = Path(temp.name)

    temp_path.replace(CACHE_PATH)


def load_bm25_index() -> bool:

    global _bm25_index
    global _bm25_chunks

    if not CACHE_PATH.exists():
        return False

    try:

        with open(
            CACHE_PATH,
            "rb",
        ) as file:

            _bm25_index, _bm25_chunks = pickle.load(file)

        return True

    except Exception:

        _bm25_index = None
        _bm25_chunks = None

        return False


def bm25_search(
    query: str,
    top_k: int = 10,
) -> List[Dict]:

    global _bm25_index
    global _bm25_chunks

    if _bm25_index is None:
        load_bm25_index()

    if _bm25_index is None:
        return []

    if not _bm25_chunks:
        return []

    hits = _bm25_index.search(
        query=query,
        top_k=top_k,
    )

    output = []

    for index, score in hits:

        if index >= len(_bm25_chunks):
            continue

        chunk = dict(_bm25_chunks[index])

        chunk["bm25_score"] = round(score, 4)

        output.append(chunk)

    return output


def get_all_chunks_from_index() -> List[Dict]:

    return list(_bm25_chunks) if _bm25_chunks else []