"""Case data loader — lets trajectory-analysis workers query raw observability data.

The analysis orchestrator dispatches workers to analyze individual cases.
Each worker can call ``load_case_data(case_id)`` to load the parquet files
for that case into DuckDB, then use ``describe_tables`` / ``query_sql``
to query the actual metrics, traces, and logs the RCA agent had access to.

Setup (called by the batch runner before the system starts)::

    from agentm.tools.case_data import set_case_data_mapping
    set_case_data_mapping({"43258": "/data/cases/ts0-...", ...})

The mapping is stored in a ``ContextVar`` so concurrent batch runs each
get their own isolated mapping.  Child asyncio tasks (workers) inherit
the mapping from their parent via the standard context-copy mechanism.
"""

from __future__ import annotations

import contextvars
import json
from pathlib import Path
from typing import Any

from agentm.tools.duckdb_sql import register_tables

# ContextVar: case_id -> data directory path.
# Using ContextVar (instead of a plain dict) ensures that concurrent
# batch runs each maintain their own mapping without interference.
_case_data_var: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "case_data_mapping", default={}
)

# Parquet files to register as DuckDB tables (same set the RCA agent uses).
_ALLOWED_PARQUET_FILES: frozenset[str] = frozenset(
    {
        "abnormal_metrics.parquet",
        "normal_metrics.parquet",
        "abnormal_metrics_histogram.parquet",
        "normal_metrics_histogram.parquet",
        "abnormal_metrics_sum.parquet",
        "normal_metrics_sum.parquet",
        "abnormal_traces.parquet",
        "normal_traces.parquet",
        "abnormal_logs.parquet",
        "normal_logs.parquet",
    }
)


def set_case_data_mapping(mapping: dict[str, str]) -> None:
    """Register the case_id -> data_dir mapping for the current context.

    Uses ``ContextVar.set`` so that each asyncio task (batch run) gets
    its own isolated copy of the mapping.
    """
    _case_data_var.set(dict(mapping))


async def load_case_data(case_id: str) -> str:
    """Load observability data (parquet files) for a specific RCA case.

    After calling this, use ``describe_tables`` to see available tables
    and ``query_sql`` to run SQL queries against the raw metrics, traces,
    and logs that the RCA agent had access to during investigation.

    This lets you verify:
    - What signals were available in the data for the ground-truth root cause
    - Whether the agent missed obvious anomalies
    - What a correct query path would have looked like

    Args:
        case_id: The case identifier (matches the trajectory thread_id / eval ID).

    Returns:
        JSON with registered table names and row counts, or an error message.
    """
    mapping = _case_data_var.get()

    if not mapping:
        return json.dumps(
            {
                "error": "No case data mapping configured.",
                "hint": "Observability data access is not available for this run. "
                "The batch config needs a data_base_dir setting.",
            }
        )

    data_dir = mapping.get(case_id)
    if data_dir is None:
        available = list(mapping.keys())[:10]
        return json.dumps(
            {
                "error": f"No data directory mapped for case_id={case_id!r}.",
                "available_case_ids": available,
            }
        )

    data_path = Path(data_dir)
    if not data_path.is_dir():
        return json.dumps(
            {"error": f"Data directory does not exist: {data_dir}"}
        )

    # Discover parquet files
    tables: dict[str, Any] = {}
    for f in sorted(data_path.iterdir()):
        if f.name in _ALLOWED_PARQUET_FILES and f.is_file():
            table_name = f.stem  # e.g. "abnormal_traces"
            tables[table_name] = str(f)

    if not tables:
        return json.dumps(
            {
                "error": f"No parquet files found in {data_dir}",
                "hint": "Expected files like abnormal_traces.parquet, abnormal_logs.parquet, etc.",
            }
        )

    # Register tables in the current async context (ContextVar isolation)
    register_tables(tables)

    return json.dumps(
        {
            "status": "ok",
            "case_id": case_id,
            "data_dir": data_dir,
            "tables": list(tables.keys()),
            "table_count": len(tables),
        }
    )
