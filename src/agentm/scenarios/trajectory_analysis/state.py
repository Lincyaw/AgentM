"""Trajectory analysis state schema."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from agentm.models.state import BaseExecutorState


class TrajectoryAnalysisState(BaseExecutorState):
    """Generic state for all trajectory analysis skills.

    Inherits messages, task_id, task_description, current_phase
    from BaseExecutorState.
    """

    source_trajectories: list[str]
    skill_name: str
    analysis_results: Annotated[list[dict], operator.add]
    structured_output: Optional[Any]
    feedback: str
