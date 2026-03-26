"""Core data types for the Agent Harness SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    """Lifecycle status of an agent."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RunConfig:
    """Per-run execution configuration."""

    max_steps: int | None = None
    timeout: float | None = None
    thread_id: str | None = None  # for checkpointing
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of an agent loop execution."""

    agent_id: str
    status: AgentStatus
    output: Any = None  # final response (str, dict, Pydantic model)
    error: str | None = None
    duration_seconds: float | None = None
    steps: int = 0  # total LLM call rounds
    tool_calls: int = 0  # total tool invocations
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvent:
    """Streaming event from an agent loop.

    Event types: llm_start, llm_end, tool_start, tool_end,
    inject, complete, error.
    """

    type: str
    agent_id: str
    data: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    timestamp: str = ""


@dataclass
class AgentInfo:
    """Runtime status snapshot of an agent."""

    agent_id: str
    status: AgentStatus
    parent_id: str | None = None
    current_step: int = 0
    started_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result: AgentResult | None = None


@dataclass(frozen=True)
class LoopContext:
    """Read-only context available to middleware during a loop iteration."""

    agent_id: str
    step: int
    max_steps: int | None
    tool_call_count: int
    metadata: dict[str, Any]
