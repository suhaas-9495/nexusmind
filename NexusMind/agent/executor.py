"""
NexusMind – Agent Executor
Maps planned steps to concrete tool functions and runs them.
"""

from typing import Dict, Any
from agent.memory import AgentStepMemory
from tools.rag_tool      import rag_retrieve
from tools.calculator    import calculate
from tools.summarizer    import summarize_results
from tools.analyzer      import analyze_results
from tools.synthesizer   import synthesize_results


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Any] = {
    "retrieve":   rag_retrieve,
    "calculate":  calculate,
    "summarize":  summarize_results,
    "analyze":    analyze_results,
    "synthesize": synthesize_results,
}


def execute_step(step: str, memory: AgentStepMemory, sources_accumulator: list) -> str:
    """
    Execute a single planned step.

    Args:
        step:                Raw step string from planner (e.g. "retrieve AI trends").
        memory:              Agent's step memory for context accumulation.
        sources_accumulator: Mutable list; rag_retrieve appends source chunks here.

    Returns:
        Result string from the tool.
    """
    step_lower = step.lower().strip()

    # ── Match tool by prefix ──────────────────────────────────────────────────
    for tool_name, tool_fn in TOOL_REGISTRY.items():
        if step_lower.startswith(tool_name):
            argument = step[len(tool_name):].strip()

            if tool_name == "retrieve":
                result, sources = tool_fn(argument)
                sources_accumulator.extend(sources)
                return result

            if tool_name in ("summarize", "analyze", "synthesize"):
                return tool_fn(memory.get_results())

            if tool_name == "calculate":
                return tool_fn(argument)

    # ── Fallback: treat entire step as a retrieve query ───────────────────────
    result, sources = rag_retrieve(step)
    sources_accumulator.extend(sources)
    return result
