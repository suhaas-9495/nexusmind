"""NexusMind – Analyzer Tool"""
from typing import List


def analyze_results(results: List[str]) -> str:
    if not results:
        return "No data to analyze."
    topics = set()
    total_chars = 0
    for r in results:
        total_chars += len(r)
        # Extract simple keyword hints
        words = r.lower().split()
        for w in words:
            if len(w) > 6 and w.isalpha():
                topics.add(w)

    top_topics = sorted(topics, key=lambda x: -len(x))[:10]
    return (
        f"Analysis of {len(results)} retrieved results "
        f"({total_chars} chars total).\n"
        f"Key themes detected: {', '.join(top_topics) or 'none'}.\n"
        f"Data appears {'substantial' if total_chars > 1000 else 'limited'} "
        f"for answering the query."
    )
