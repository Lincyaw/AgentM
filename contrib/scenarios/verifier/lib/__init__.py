"""Verifier shared utilities."""
from __future__ import annotations

import os
from pathlib import Path


def duckdb_conn(data_dir: Path):  # noqa: ANN202
    """Create an in-memory DuckDB connection with views over all parquet files."""
    import duckdb

    conn = duckdb.connect(":memory:")
    cap = os.environ.get("AGENTM_DUCKDB_THREADS")
    if cap:
        try:
            conn.execute(f"SET threads={max(1, int(cap))}")
        except (ValueError, duckdb.Error):
            pass
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            conn.execute(
                f"CREATE VIEW {f.stem} AS "
                f"SELECT * FROM read_parquet('{f.as_posix()}')"
            )
    return conn
