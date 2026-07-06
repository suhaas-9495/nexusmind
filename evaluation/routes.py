"""
NexusMind v2 — /evaluate endpoint
Trigger evaluation and fetch results via API.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

from auth.jwt_handler import get_current_user
from observability.metrics import get_summary

router = APIRouter()


class EvalRequest(BaseModel):
    k_values: List[int] = [1, 3, 5, 10]
    custom_dataset: Optional[List[Dict]] = None


@router.post("/run")
async def run_evaluation(
    req: EvalRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run BM25 vs Dense vs Hybrid evaluation."""
    try:
        from evaluation.runner import run_evaluation as _run, EVAL_DATASET
        dataset = req.custom_dataset or EVAL_DATASET
        results = _run(dataset=dataset, k_values=req.k_values)
        # Strip per_query detail for API response
        return {k: {m: v for m, v in v.items() if m != "per_query"}
                if isinstance(v, dict) else v
                for k, v in results.items()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Return live latency/usage metrics."""
    return get_summary()
