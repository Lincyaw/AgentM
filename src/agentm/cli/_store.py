"""Host-side trajectory storage resolution for CLI commands."""

from __future__ import annotations

from agentm.storage.trajectory.resolve import (
    ResolvedTrajectoryStorage,
    resolve_trajectory_storage,
    resolve_trajectory_storage_or_create,
)

__all__ = [
    "ResolvedTrajectoryStorage",
    "resolve_trajectory_storage",
    "resolve_trajectory_storage_or_create",
]
