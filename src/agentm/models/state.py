"""State schemas (TypedDict) for AgentM agent systems.

SDK state types are defined here. Domain-specific states live in their
canonical locations under ``scenarios/``.

The global TypeVar ``S`` allows framework components (builder, middleware,
task manager) to be generic over user-defined state types that extend
``BaseExecutorState``.
"""

from __future__ import annotations

from typing import Annotated, TypedDict, TypeVar

from langgraph.graph.message import add_messages

from agentm.models.data import CompressionRef


class BaseExecutorState(TypedDict):
    """Fields shared by all agent systems.

    ``current_phase`` is a plain ``str`` so the framework layer stays
    domain-agnostic.  Concrete strategies use their own phase enums
    internally (e.g. ``Phase`` for RCA) and convert to ``str`` at the
    boundary.
    """

    messages: Annotated[list, add_messages]
    task_id: str
    task_description: str
    current_phase: str


# Global TypeVar for generic framework components.
S = TypeVar("S", bound=BaseExecutorState)


class SubAgentState(TypedDict):
    """State for independently compiled Sub-Agent subgraphs."""

    messages: Annotated[list, add_messages]
    scratchpad: list[str]
    observations: list[str]
    tool_call_count: int
    compression_refs: list[CompressionRef]
