"""Resolve the trajectory store for CLI sessions.

Priority:
1. AGENTM_TRAJECTORY_DSN -> Postgres
2. AGENTM_TRAJECTORY_DIR -> JSONL directory
3. $CWD/.agentm/trajectory/ -> JSONL (project-local)
4. None (no configured persistence)

An explicitly configured backend is fail-fast. In particular, a Postgres DSN
never degrades to JSONL when the optional driver is missing or connection
setup fails.
"""

from __future__ import annotations

import os
from pathlib import Path

from agentm.core.abi.store import TrajectoryStore


def resolve_trajectory_store() -> TrajectoryStore | None:
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


def resolve_trajectory_store_or_create() -> TrajectoryStore:
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


def _postgres_store(dsn: str) -> TrajectoryStore:
    try:
        from agentm.storage.trajectory.psycopg import connect
    except ModuleNotFoundError as exc:
        if exc.name not in {"psycopg", "psycopg_binary"}:
            raise
        raise RuntimeError(
            "Postgres trajectory storage requires the "
            "'agentm[storage-postgres]' extra"
        ) from exc

    from agentm.storage.trajectory.postgres_turns import PostgresTrajectoryStore

    conn = connect(dsn)
    return PostgresTrajectoryStore(conn, create_schema=True)


def _jsonl_store(directory: Path) -> TrajectoryStore:
    from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore

    directory.mkdir(parents=True, exist_ok=True)
    return JsonlTrajectoryStore(directory)
