"""
NexusMind – Agent Runner
Orchestrates: plan → execute → synthesize → answer.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

from agent.state    import AgentState
from agent.planner  import create_plan
from agent.executor import execute_step
from agent.memory   import AgentStepMemory, ConversationMemory
from backend.config import AGENT_MAX_STEPS, AGENT_MAX_RETRIES, AGENT_TIMEOUT_SEC
from backend.llm_answer import generate_answer
from backend.embedding  import embed_texts
from backend.retrieval  import retrieve_similar_chunks

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _log_event(event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG_DIR / "agent_logs.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


def run_agent(
    user_query: str,
    conversation_memory: ConversationMemory | None = None,
) -> Dict[str, Any]:
    """
    Full agentic RAG run.

    Returns a dict:
        {
            "final_answer": str,
            "sources":      list[dict],
            "steps":        list[str],   # human-readable step log
            "success":      bool,
        }
    """
    state   = AgentState(user_query=user_query)
    memory  = AgentStepMemory()
    sources: list = []

    start_time = time.time()

    # ── 1. Planning ───────────────────────────────────────────────────────────
    state.plan = create_plan(user_query)[:AGENT_MAX_STEPS]
    state.log(f"📋 Plan created: {len(state.plan)} steps")
    _log_event({"type": "plan", "query": user_query, "plan": state.plan})

    # ── 2. Execution loop ─────────────────────────────────────────────────────
    for step in state.plan:

        # Timeout guard
        if time.time() - start_time > AGENT_TIMEOUT_SEC:
            state.log("⏰ Timeout reached — stopping early.")
            break

        attempts = 0
        result   = "Step failed"

        while attempts <= AGENT_MAX_RETRIES:
            try:
                result  = execute_step(step, memory, sources)
                break
            except Exception as exc:
                attempts += 1
                logger.warning(f"Step '{step}' attempt {attempts} failed: {exc}")
                if attempts > AGENT_MAX_RETRIES:
                    result = f"[Error after {AGENT_MAX_RETRIES} retries: {exc}]"

        # Truncate huge results
        if len(str(result)) > 1500:
            result = str(result)[:1500] + "…[truncated]"

        memory.add(step, result)
        state.tool_results[step] = result
        state.current_step      += 1
        state.log(f"✅ Step {state.current_step}: `{step}` → {str(result)[:120]}")
        _log_event({"type": "step", "step": step, "result": result})

    # ── 3. Final answer synthesis ─────────────────────────────────────────────
    conv_ctx = conversation_memory.get_context() if conversation_memory else ""

    # Deduplicate sources by chunk_index + source filename
    seen    = set()
    unique_sources = []
    for src in sources:
        key = (src.get("source"), src.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    if unique_sources:
        try:
            final_answer = generate_answer(user_query, unique_sources[:6], conv_ctx)
        except Exception as e:
            # Fallback: stitch step results together
            final_answer = "\n\n".join(memory.get_results())
            state.log(f"⚠️ LLM synthesis failed ({e}), using step results.")
    else:
        # No retrieved chunks — ask LLM with just conversation context
        final_answer = (
            "I couldn't find relevant information in the indexed documents. "
            "Please make sure the relevant document is uploaded and indexed."
        )

    state.final_answer = final_answer
    state.sources      = unique_sources
    state.success      = True

    elapsed = round(time.time() - start_time, 2)
    state.log(f"🏁 Done in {elapsed}s")
    _log_event({
        "type":    "complete",
        "elapsed": elapsed,
        "steps":   state.current_step,
    })

    return {
        "final_answer": state.final_answer,
        "sources":      state.sources,
        "steps":        state.steps_log,
        "success":      state.success,
    }
