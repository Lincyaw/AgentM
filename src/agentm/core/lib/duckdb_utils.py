"""DuckDB connection utilities shared across scenarios."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

_ENV_DUCKDB_THREADS = "AGENTM_DUCKDB_THREADS"


def cap_duckdb_threads(conn: Any) -> None:
    """Cap DuckDB thread count via ``AGENTM_DUCKDB_THREADS`` env var.

    Each ``duckdb.connect()`` defaults its task scheduler to the host core
    count.  When many agent subprocesses run concurrently (e.g. eval
    fan-out), every connection grabbing all cores oversubscribes the box.
    The data is tiny (per-case parquet), so a low cap costs no real query
    speed.  No-op when the env var is unset; preserves DuckDB's default.
    """
    raw = os.environ.get(_ENV_DUCKDB_THREADS)
    if not raw:
        return
    try:
        n = max(1, int(raw))
    except ValueError:
        return
    try:
        conn.execute(f"SET threads={n}")
    except Exception:  # noqa: BLE001
        logger.debug("cap_duckdb_threads: SET threads={} failed", n)
