"""NexusMind – Synthesizer Tool"""
from typing import List


def synthesize_results(results: List[str]) -> str:
    if not results:
        return "No results available to synthesize."
    parts = [f"**Finding {i+1}:** {r.strip()[:500]}" for i, r in enumerate(results) if r.strip()]
    return "Synthesized findings:\n\n" + "\n\n".join(parts)
