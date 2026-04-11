"""Unit tests for DebugConfig and SystemConfig defaults."""

from __future__ import annotations

from agentm.config.schema import (
    DebugConfig,
    StorageBackendConfig,
    StorageConfig,
    SystemConfig,
    TrajectoryConfig,
)


def _base_system_config(**overrides: object) -> SystemConfig:
    payload = {
        "models": {},
        "storage": StorageConfig(
            checkpointer=StorageBackendConfig(backend="memory", url=""),
            store=StorageBackendConfig(backend="memory", url=""),
        ),
        "recovery": {"mode": "manual", "expose_api": True},
    }
    payload.update(overrides)
    return SystemConfig(**payload)


def test_system_config_supports_debug_override_and_safe_defaults() -> None:
    custom = _base_system_config(
        debug=DebugConfig(
            trajectory=TrajectoryConfig(enabled=True, output_dir="/tmp/traj"),
            console_live=True,
            verbose=True,
        )
    )
    assert custom.debug.trajectory.enabled is True
    assert custom.debug.console_live is True
    assert custom.debug.trajectory.output_dir == "/tmp/traj"

    defaults = _base_system_config()
    assert defaults.debug.console_live is False
    assert defaults.debug.verbose is False
    assert defaults.debug.trajectory.enabled is True
    assert defaults.debug.trajectory.output_dir == "./trajectories"
