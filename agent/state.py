"""
NexusMind – Agent State
Tracks the lifecycle of a single agent run.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AgentState:
    user_query:    str
    plan:          List[str]       = field(default_factory=list)
    current_step:  int             = 0
    tool_results:  Dict[str, str]  = field(default_factory=dict)
    sources:       List[Dict]      = field(default_factory=list)
    final_answer:  str             = ""
    steps_log:     List[str]       = field(default_factory=list)
    success:       bool            = False

    def log(self, msg: str) -> None:
        self.steps_log.append(msg)
