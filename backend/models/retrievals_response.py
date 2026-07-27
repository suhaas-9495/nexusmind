from pydantic import BaseModel
from typing import List, Dict


class RetrievalResponse(BaseModel):

    results: List[Dict]

    semantic_hits: int

    bm25_hits: int

    fusion_method: str
    
