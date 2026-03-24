"""Shared typing aliases across AgentM."""

from __future__ import annotations

from typing import TypeAlias

# str so that trajectory-analysis task types (collect, analyze, extract, refine)
# flow through unmodified alongside the existing RCA types.
TaskType: TypeAlias = str
