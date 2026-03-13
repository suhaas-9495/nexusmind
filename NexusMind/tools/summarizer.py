"""NexusMind – Summarizer Tool"""
from typing import List


def summarize_results(results: List[str]) -> str:
    if not results:
        return "Nothing to summarize yet."
    combined = "\n\n".join(str(r) for r in results if r)
    # Truncate for LLM context
    if len(combined) > 4000:
        combined = combined[:4000] + "…"
    return f"Summary of {len(results)} steps:\n\n{combined}"
