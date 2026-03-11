"""State schemas (TypedDict) for AgentM agent systems.

Normative definitions from design docs. Field names and types are binding.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages

from agentm.models.data import (
    CompressionRef,
    DiagnosticNotebook,
    KnowledgeEntry,
)
from agentm.models.enums import Phase


class BaseExecutorState(TypedDict):
    """Fields shared by all agent systems."""

    messages: Annotated[list, add_messages]
    task_id: str
    task_description: str
    current_phase: Phase


class SubAgentState(TypedDict):
    """State for independently compiled Sub-Agent subgraphs."""

    messages: Annotated[list, add_messages]
    scratchpad: list[str]
    observations: list[str]
    tool_call_count: int
    compression_refs: list[CompressionRef]


class HypothesisDrivenState(BaseExecutorState):
    """Hypothesis-driven RCA state.

    Primary state schema for the hypothesis-driven RCA Orchestrator.
    Inherits messages, task_id, task_description, current_phase from BaseExecutorState.
    """

    notebook: DiagnosticNotebook
    current_hypothesis: Optional[str]
    compression_refs: list[CompressionRef]
    structured_response: Optional[Any]


class SequentialDiagnosisState(BaseExecutorState):
    """Sequential step-by-step diagnosis state."""

    steps: Annotated[list[dict], operator.add]
    current_step_index: int


class MemoryExtractionState(BaseExecutorState):
    """Cross-task knowledge extraction state."""

    source_trajectories: list[str]
    extracted_patterns: Annotated[list[dict], operator.add]
    knowledge_entries: list[KnowledgeEntry]
    existing_knowledge: list[KnowledgeEntry]


class DecisionTreeState(BaseExecutorState):
    """Decision tree classification state."""

    decision_path: list[str]
    current_node_id: str
    feature_values: dict[str, Any]
