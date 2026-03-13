"""RCA-specific state schema."""

from __future__ import annotations

from typing import Any, Optional

from agentm.models.data import CompressionRef
from agentm.scenarios.rca.data import DiagnosticNotebook
from agentm.models.state import BaseExecutorState


class HypothesisDrivenState(BaseExecutorState):
    """Hypothesis-driven RCA state.

    Primary state schema for the hypothesis-driven RCA Orchestrator.
    Inherits messages, task_id, task_description, current_phase from BaseExecutorState.
    """

    notebook: DiagnosticNotebook
    current_hypothesis: Optional[str]
    compression_refs: list[CompressionRef]
    structured_response: Optional[Any]
