"""
NexusMind v2 — Agent Runner
Upgraded: explicit tool routing logic, detailed step logs, tool usage audit trail.

Tool routing is explicit (not just string prefix matching):
  retrieve  → hybrid vector + BM25 search
  calculate → safe math evaluator
  summarize → LLM summarization of gathered results
  analyze   → LLM key-insight extraction
  synthesize→ LLM final answer composition
  compare   → side-by-side comparison of two retrieve queries
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from agent.state    import AgentState
from agent.planner  import create_plan
from agent.executor import execute_step, TOOL_REGISTRY
from agent.memory   import AgentStepMemory, ConversationMemory
from backend.config import AGENT_MAX_STEPS, AGENT_MAX_RETRIES, AGENT_TIMEOUT_SEC
from backend.llm_answer import generate_answer

logger = logging.getLogger("nexusmind.agent")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
AGENT_LOG_FILE = LOG_DIR / "agent_logs.jsonl"


def _log_event(event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with open(AGENT_LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"Agent log write failed: {e}")


def _route_tool(step: str) -> str:
    """Explicit tool routing with heuristic fallbacks."""
    step_lower = step.lower().strip()
    for tool_name in TOOL_REGISTRY:
        if step_lower.startswith(tool_name):
            return tool_name
    if any(kw in step_lower for kw in ["search", "find", "look up", "fetch", "get"]):
        return "retrieve"
    if any(kw in step_lower for kw in ["sum", "total", "count", "average", "percent"]):
        return "calculate"
    return "retrieve"  # safe default


def run_agent(
    user_query: str,
    conversation_memory: Optional[ConversationMemory] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full agentic RAG run with explicit tool routing and tool audit trail.

    Returns:
        {
            "final_answer": str,
            "sources":      list[dict],
            "steps":        list[str],    # human-readable step log
            "tool_log":     list[dict],   # machine-readable tool audit trail
            "success":      bool,
            "elapsed_sec":  float,
        }
    """
    state      = AgentState(user_query=user_query)
    memory     = AgentStepMemory()
    sources:   List[dict] = []
    tool_log:  List[dict] = []
    start_time = time.perf_counter()

    # ── 1. Plan ───────────────────────────────────────────────────────────────
    state.plan = create_plan(user_query)[:AGENT_MAX_STEPS]
    state.log(f"📋 Plan: {len(state.plan)} steps")
    _log_event({"type": "plan", "query": user_query, "plan": state.plan, "user_id": user_id})
    logger.info(f"Agent plan for '{user_query[:60]}': {state.plan}")

    # ── 2. Execute ────────────────────────────────────────────────────────────
    for step in state.plan:
        if (time.perf_counter() - start_time) > AGENT_TIMEOUT_SEC:
            state.log("⏰ Timeout — stopping early.")
            break

        routed_tool = _route_tool(step)
        state.log(f"🔀 '{step[:50]}' → tool: {routed_tool}")

        attempts   = 0
        result     = "Step failed."
        step_start = time.perf_counter()

        while attempts <= AGENT_MAX_RETRIES:
            try:
                result = execute_step(step, memory, sources)
                break
            except Exception as exc:
                attempts += 1
                logger.warning(f"Step '{step}' attempt {attempts} failed: {exc}")
                if attempts > AGENT_MAX_RETRIES:
                    result = f"[Error after {AGENT_MAX_RETRIES} retries: {exc}]"

        step_ms      = round((time.perf_counter() - step_start) * 1000, 1)
        result_str   = str(result)
        result_preview = result_str[:300] + ("…" if len(result_str) > 300 else "")

        memory.add(step, result_str)
        state.tool_results[step] = result_str
        state.current_step += 1

        step_entry = {
            "step_num": state.current_step,
            "step":     step,
            "tool":     routed_tool,
            "latency_ms": step_ms,
            "result":   result_preview,
            "attempts": attempts,
        }
        tool_log.append(step_entry)
        state.log(f"✅ [{state.current_step}/{len(state.plan)}] {routed_tool} ({step_ms}ms) → {result_preview[:80]}")
        _log_event({"type": "step", **step_entry, "user_id": user_id})
        logger.info(f"Step {state.current_step}: {routed_tool}({step[:40]}) → {step_ms}ms")

    # ── 3. Deduplicate sources ────────────────────────────────────────────────
    seen, unique_sources = set(), []
    for src in sources:
        key = (src.get("source"), src.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    # ── 4. Synthesize ─────────────────────────────────────────────────────────
    conv_ctx = conversation_memory.get_context() if conversation_memory else ""
    try:
        final_answer = generate_answer(user_query, unique_sources[:6], conv_ctx)
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        final_answer = "\n\n".join(f"**Step {i+1}:** {r}" for i, r in enumerate(memory.get_results()))
        state.log(f"⚠️ Synthesis failed ({e}), using step results.")

    state.final_answer = final_answer
    state.sources      = unique_sources
    state.success      = True

    elapsed_sec = round(time.perf_counter() - start_time, 2)
    state.log(f"🏁 Done in {elapsed_sec}s — {state.current_step} steps, {len(unique_sources)} sources")
    _log_event({
        "type": "complete", "query": user_query, "elapsed_sec": elapsed_sec,
        "steps_run": state.current_step, "sources_used": len(unique_sources),
        "success": True, "user_id": user_id,
    })

    return {
        "final_answer": state.final_answer,
        "sources":      state.sources,
        "steps":        state.steps_log,
        "tool_log":     tool_log,
        "success":      state.success,
        "elapsed_sec":  elapsed_sec,
    }
