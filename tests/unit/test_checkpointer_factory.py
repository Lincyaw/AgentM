"""Unit tests for DebugConfig and SystemConfig."""

from __future__ import annotations

from agentm.config.schema import (
    DebugConfig,
    StorageBackendConfig,
    StorageConfig,
    SystemConfig,
    TrajectoryConfig,
)


def test_system_config_accepts_debug_section() -> None:
    """Adding DebugConfig must not break existing config loading."""
    config = SystemConfig(
        models={},
        storage=StorageConfig(
            checkpointer=StorageBackendConfig(backend="memory", url=""),
            store=StorageBackendConfig(backend="memory", url=""),
        ),
        recovery={"mode": "manual", "expose_api": True},
        debug=DebugConfig(
            trajectory=TrajectoryConfig(enabled=True, output_dir="/tmp/traj"),
            console_live=True,
            verbose=True,
        ),
    )
    assert config.debug.trajectory.enabled is True
    assert config.debug.console_live is True
    assert config.debug.trajectory.output_dir == "/tmp/traj"


def test_debug_defaults_to_non_intrusive() -> None:
    """Debug infra must default to disabled console, enabled trajectory -- no prod impact."""
    config = SystemConfig(
        models={},
        storage=StorageConfig(
            checkpointer=StorageBackendConfig(backend="memory", url=""),
            store=StorageBackendConfig(backend="memory", url=""),
        ),
        recovery={"mode": "manual", "expose_api": True},
    )
    assert config.debug.console_live is False
    assert config.debug.verbose is False
    assert config.debug.trajectory.enabled is True
    assert config.debug.trajectory.output_dir == "./trajectories"
