"""Unit tests for checkpointer factory and DebugConfig."""

from __future__ import annotations

import pytest

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
    """Debug infra must default to disabled console, enabled trajectory — no prod impact."""
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


def test_memory_backend_returns_memory_saver() -> None:
    """Default 'memory' backend must return MemorySaver for backward compat."""
    from agentm.builder import _create_checkpointer
    from langgraph.checkpoint.memory import MemorySaver

    storage = StorageConfig(
        checkpointer=StorageBackendConfig(backend="memory", url=""),
        store=StorageBackendConfig(backend="memory", url=""),
    )
    saver = _create_checkpointer(storage)
    assert isinstance(saver, MemorySaver)


def test_sqlite_backend_returns_sqlite_saver(tmp_path) -> None:
    """SQLite backend must create a SqliteSaver instance."""
    from agentm.builder import _create_checkpointer
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = str(tmp_path / "test.db")
    storage = StorageConfig(
        checkpointer=StorageBackendConfig(backend="sqlite", url=db_path),
        store=StorageBackendConfig(backend="memory", url=""),
    )
    saver = _create_checkpointer(storage)
    assert isinstance(saver, SqliteSaver)


def test_unknown_backend_returns_none() -> None:
    """Unknown backend must return None, not crash — graceful fallback."""
    from agentm.builder import _create_checkpointer

    storage = StorageConfig(
        checkpointer=StorageBackendConfig(backend="redis", url="redis://localhost"),
        store=StorageBackendConfig(backend="memory", url=""),
    )
    saver = _create_checkpointer(storage)
    assert saver is None
