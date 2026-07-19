"""Resolve the trajectory store for CLI sessions.

Delegates to ``agentm.core.lib.store_resolve`` — the canonical resolution
lives in core so that SDK-only callers (harbor, tests) get the same default.
"""

from __future__ import annotations

from agentm.core.abi.store import TrajectoryStore
from agentm.core.lib.store_resolve import (
    resolve_trajectory_store,
    resolve_trajectory_store_or_create,
)

__all__ = [
    "resolve_trajectory_store",
    "resolve_trajectory_store_or_create",
]
