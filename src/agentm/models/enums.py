"""Enumeration types for AgentM.

SDK enums are defined here. Domain-specific enums live in their
canonical locations under ``scenarios/``.
"""

from __future__ import annotations

from enum import Enum

# --- SDK enums ---


class AgentRunStatus(str, Enum):
    """Runtime status of a Sub-Agent execution (used by TaskManager)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status of a dispatched task in PendingTask."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"

