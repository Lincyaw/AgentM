"""Resolve the trajectory store for CLI sessions.

Uses ``AGENTM_TRAJECTORY_DSN`` or the local dev Postgres default.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

_DEFAULT_DSN = "postgresql://agentm:agentm@localhost:55432/agentm_test"


def resolve_trajectory_store() -> Any:
    """Return a PostgresTrajectoryStore or None."""
    return _postgres_store(os.environ.get("AGENTM_TRAJECTORY_DSN", _DEFAULT_DSN))


def resolve_trajectory_store_or_create() -> Any:
    """Same as resolve_trajectory_store (kept for CLI callers)."""
    return _postgres_store(os.environ.get("AGENTM_TRAJECTORY_DSN", _DEFAULT_DSN))


def _postgres_store(dsn: str) -> Any:
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed; trajectory persistence disabled")
        return None

    from agentm.storage.trajectory.postgres_turns import PostgresTrajectoryStore

    try:
        conn = psycopg2.connect(dsn)
        return PostgresTrajectoryStore(conn, create_schema=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("trajectory store: Postgres unavailable ({})", exc)
        return None
