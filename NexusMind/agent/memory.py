"""
NexusMind – Conversation Memory
Maintains short-term multi-turn dialogue history.
"""

from typing import List, Dict
from collections import deque


class ConversationMemory:
    """
    Ring-buffer conversation memory.
    Stores the last `max_turns` user/assistant pairs.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: deque[Dict] = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str) -> None:
        """Add a single turn (role = 'user' | 'assistant')."""
        self._history.append({"role": role, "content": content})

    def get_all(self) -> List[Dict]:
        return list(self._history)

    def get_context(self, max_chars: int = 3000) -> str:
        """
        Return a flat text summary of recent conversation,
        truncated to `max_chars` (prevents context overflow).
        """
        lines = []
        for turn in self._history:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = "…" + text[-max_chars:]
        return text

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


class AgentStepMemory:
    """
    Memory for a single agent run — stores (step, result) pairs.
    """

    def __init__(self):
        self._steps: List[Dict] = []

    def add(self, step: str, result: str) -> None:
        self._steps.append({"step": step, "result": result})

    def get_all(self) -> List[Dict]:
        return list(self._steps)

    def get_results(self) -> List[str]:
        return [s["result"] for s in self._steps]

    def clear(self) -> None:
        self._steps.clear()
