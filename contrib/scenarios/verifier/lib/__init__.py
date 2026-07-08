"""Verifier shared utilities."""
from __future__ import annotations

from pathlib import Path

from agentm.core.lib import cap_duckdb_threads


def duckdb_conn(data_dir: Path):  # noqa: ANN202
    """Create an in-memory DuckDB connection with views over all parquet files."""
    import duckdb

    conn = duckdb.connect(":memory:")
    cap_duckdb_threads(conn)
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            conn.execute(
                f"CREATE VIEW {f.stem} AS "
                f"SELECT * FROM read_parquet('{f.as_posix()}')"
            )
    return conn
