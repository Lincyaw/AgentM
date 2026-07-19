"""Resolve the trajectory store for CLI sessions.

Priority:
1. AGENTM_TRAJECTORY_DSN -> Postgres
2. AGENTM_TRAJECTORY_DIR -> JSONL directory
3. $CWD/.agentm/trajectory/ -> JSONL (project-local)
4. None (no persistence)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger


def resolve_trajectory_store() -> Any:
    """Return a TrajectoryStore or None if no backend is configured."""
    dsn = os.environ.get("AGENTM_TRAJECTORY_DSN")
    if dsn:
        return _postgres_store(dsn)

    explicit_dir = os.environ.get("AGENTM_TRAJECTORY_DIR")
    if explicit_dir:
        return _jsonl_store(Path(explicit_dir).expanduser())

    project_local = Path.cwd() / ".agentm" / "trajectory"
    if project_local.is_dir():
        return _jsonl_store(project_local)

    return None


def resolve_trajectory_store_or_create() -> Any:
    """Like resolve_trajectory_store, but auto-creates project-local dir."""
    dsn = os.environ.get("AGENTM_TRAJECTORY_DSN")
    if dsn:
        return _postgres_store(dsn)

    explicit_dir = os.environ.get("AGENTM_TRAJECTORY_DIR")
    if explicit_dir:
        return _jsonl_store(Path(explicit_dir).expanduser())

    project_local = Path.cwd() / ".agentm" / "trajectory"
    project_local.mkdir(parents=True, exist_ok=True)
    return _jsonl_store(project_local)


def _postgres_store(dsn: str) -> Any:
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        logger.warning("psycopg2 not installed; trajectory persistence disabled")
        return None

    from agentm.storage.trajectory.postgres_turns import PostgresTrajectoryStore

    conn = psycopg2.connect(dsn)
    return PostgresTrajectoryStore(conn, create_schema=True)


def _jsonl_store(directory: Path) -> Any:
    from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore

    directory.mkdir(parents=True, exist_ok=True)
    return JsonlTrajectoryStore(directory)
