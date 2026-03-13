"""
NexusMind – BM25 Index
Builds and queries a BM25 keyword index over all stored chunk texts.
BM25 (Best Match 25) is the gold-standard sparse retrieval algorithm —
it excels at exact keyword matches, rare terms, and named entities that
semantic search often misses.

The index is rebuilt from Qdrant payloads each session (lightweight for
document-scale data). For very large corpora, persist to disk instead.
"""

import re
import math
import pickle
from pathlib import Path
from typing import List, Dict, Tuple


# ── Simple BM25 implementation (no external dependency needed) ────────────────

class BM25:
    """
    BM25Okapi implementation from scratch.
    k1=1.5, b=0.75 are standard defaults used by Elasticsearch/Lucene.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        # Populated on fit()
        self.corpus_size:   int               = 0
        self.avgdl:         float             = 0.0
        self.doc_freqs:     List[Dict]        = []   # term → freq per doc
        self.idf:           Dict[str, float]  = {}   # term → idf score
        self.doc_len:       List[int]         = []   # token count per doc
        self._corpus_texts: List[str]         = []   # raw texts (for inspection)

    # ── Tokeniser ─────────────────────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens (noise)
        return [t for t in tokens if len(t) > 1]

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, corpus: List[str]) -> None:
        """Build BM25 index from a list of document strings."""
        self._corpus_texts = corpus
        self.corpus_size   = len(corpus)

        tokenized = [self.tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in tokenized]
        self.avgdl   = sum(self.doc_len) / max(self.corpus_size, 1)

        # Term frequencies per document
        self.doc_freqs = []
        df: Dict[str, int] = {}   # document frequency

        for tokens in tokenized:
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        # IDF with smoothing (Robertson IDF)
        self.idf = {}
        for term, doc_freq in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1
            )

    # ── Score ─────────────────────────────────────────────────────────────────

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single document."""
        query_tokens = self.tokenize(query)
        freq_map     = self.doc_freqs[doc_idx]
        dl           = self.doc_len[doc_idx]
        score        = 0.0

        for term in query_tokens:
            if term not in freq_map:
                continue
            tf  = freq_map[term]
            idf = self.idf.get(term, 0.0)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score      += idf * (numerator / denominator)

        return score

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_idx, bm25_score) sorted best-first.
        """
        if self.corpus_size == 0:
            return []

        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: -x[1])
        # Filter zero-score results (no keyword overlap at all)
        return [(idx, s) for idx, s in scores[:top_k] if s > 0.0]


# ── Index manager (singleton) ─────────────────────────────────────────────────

_bm25_index:  BM25       | None = None
_bm25_chunks: List[Dict] | None = None   # parallel list of chunk payloads

CACHE_PATH = Path("logs/bm25_cache.pkl")


def build_bm25_index(chunks: List[Dict]) -> None:
    """
    Build (or rebuild) the in-memory BM25 index from a list of chunk dicts.
    Each dict must have a 'chunk_text' key.
    Also persists to disk so it survives a Streamlit rerun.
    """
    global _bm25_index, _bm25_chunks

    texts = [c.get("chunk_text", "") for c in chunks]

    _bm25_index = BM25()
    _bm25_index.fit(texts)
    _bm25_chunks = list(chunks)

    # Persist
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((_bm25_index, _bm25_chunks), f)


def load_bm25_index() -> bool:
    """Load persisted BM25 index from disk. Returns True if successful."""
    global _bm25_index, _bm25_chunks
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                _bm25_index, _bm25_chunks = pickle.load(f)
            return True
        except Exception:
            pass
    return False


def bm25_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Search the BM25 index.
    Returns list of chunk dicts with an added 'bm25_score' key.
    """
    global _bm25_index, _bm25_chunks

    # Try loading from disk if not in memory
    if _bm25_index is None:
        load_bm25_index()

    if _bm25_index is None or not _bm25_chunks:
        return []

    hits = _bm25_index.search(query, top_k=top_k)

    results = []
    for idx, score in hits:
        if idx < len(_bm25_chunks):
            chunk = dict(_bm25_chunks[idx])
            chunk["bm25_score"] = round(score, 4)
            results.append(chunk)

    return results


def get_all_chunks_from_index() -> List[Dict]:
    """Return all chunks stored in the BM25 index (used for rebuilding)."""
    return list(_bm25_chunks) if _bm25_chunks else []
