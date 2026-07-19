"""Environment lifecycle backends for AgentM hosts."""

from agentm.environments.local import (
    LocalEnvironmentOperations,
    LocalSnapshotEffectScope,
    LocalSnapshotStore,
)

__all__ = [
    "LocalEnvironmentOperations",
    "LocalSnapshotEffectScope",
    "LocalSnapshotStore",
]
