"""
NexusMind – Agent Planner
Uses the local Ollama LLM to break a user query into executable steps.
"""

import os
import requests
from typing import List

from backend.config import OLLAMA_BASE_URL, LLM_MODEL

PLANNER_SYSTEM = """You are an expert AI planning agent for a document Q&A system.

Given a user query, produce a numbered step-by-step plan.

Available tools the executor can call:
  - retrieve <query>     : searches the vector store for relevant document chunks
  - calculate <expr>     : evaluates a math expression (e.g. "calculate 42 * 3")
  - summarize            : summarizes all retrieved results so far
  - analyze              : analyzes and extracts key insights from retrieved data
  - synthesize           : combines multiple retrieved chunks into a cohesive answer

Rules:
  - Output ONLY numbered steps, one per line (e.g. "1. retrieve key findings about X")
  - Each step must start with a tool name or be a reasoning step
  - Do NOT explain yourself or add commentary
  - Keep steps minimal and purposeful (3-6 steps max)
  - Always end with a synthesize or summarize step
"""


def create_plan(user_query: str) -> List[str]:
    """
    Ask the LLM to decompose `user_query` into an ordered list of steps.
    Falls back to a simple 3-step plan if the LLM call fails.
    """
    model = os.environ.get("LLM_MODEL", LLM_MODEL)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user",   "content": f"User query: {user_query}"},
    ]

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]

        steps = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Accept lines starting with a digit+dot or digit+)
            if line and line[0].isdigit():
                # Strip "1. " or "1) "
                parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
                if len(parts) == 2:
                    steps.append(parts[1].strip())
                else:
                    steps.append(line)

        if steps:
            return steps

    except Exception:
        pass

    # Fallback plan
    return [
        f"retrieve {user_query}",
        "analyze",
        "synthesize",
    ]
