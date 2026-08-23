"""Environment lifecycle backends for AgentM hosts."""

from agentm.environments.local import (
    LocalBashOperations,
    LocalEnvironmentOperations,
    LocalSnapshotEffectScope,
    LocalSnapshotStore,
)

__all__ = [
    "LocalBashOperations",
    "LocalEnvironmentOperations",
    "LocalSnapshotEffectScope",
    "LocalSnapshotStore",
]
