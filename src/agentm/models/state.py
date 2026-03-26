"""State schemas (TypedDict) for AgentM agent systems.

SDK state types are defined here. Domain-specific states live in their
canonical locations under ``scenarios/``.

No LangGraph dependency — these are plain TypedDict schemas used as
mutable dict templates by strategies and format_context functions.
"""

from __future__ import annotations

from typing import TypedDict


class BaseExecutorState(TypedDict):
    """Fields shared by all agent systems.

    ``current_phase`` is a plain ``str`` so the framework layer stays
    domain-agnostic.  Concrete strategies use their own phase enums
    internally (e.g. ``Phase`` for RCA) and convert to ``str`` at the
    boundary.
    """

    messages: list
    task_id: str
    task_description: str
    current_phase: str
