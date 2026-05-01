from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentm_rca.tools import close_obs_connection


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


@pytest.fixture
def observability_data_dir(tmp_path: Path) -> Path:
    ts0 = datetime(2025, 8, 28, 20, 45)
    ts1 = datetime(2025, 8, 28, 20, 47)

    _write_table(
        tmp_path / "abnormal_metrics.parquet",
        [
            {
                "time": ts0,
                "metric": "cpu.utilization",
                "value": 1.0,
                "service_name": "checkout",
                "attr.k8s.deployment.name": "checkout-v1",
                "attr.k8s.pod.name": "checkout-abc",
                "attr.k8s.node.name": "node-a",
                "attr.k8s.namespace.name": "prod",
            },
            {
                "time": ts1,
                "metric": "cpu.utilization",
                "value": 3.0,
                "service_name": "checkout",
                "attr.k8s.deployment.name": "checkout-v1",
                "attr.k8s.pod.name": "checkout-abc",
                "attr.k8s.node.name": "node-a",
                "attr.k8s.namespace.name": "prod",
            },
        ],
    )
    _write_table(
        tmp_path / "normal_metrics.parquet",
        [
            {
                "time": ts0,
                "metric": "cpu.utilization",
                "value": 0.5,
                "service_name": "checkout",
                "attr.k8s.deployment.name": "checkout-v1",
                "attr.k8s.pod.name": "checkout-abc",
                "attr.k8s.node.name": "node-a",
                "attr.k8s.namespace.name": "prod",
            }
        ],
    )
    _write_table(
        tmp_path / "abnormal_traces.parquet",
        [
            {
                "time": ts0,
                "trace_id": "trace-1",
                "span_id": "parent",
                "parent_span_id": None,
                "span_name": "GET /checkout",
                "service_name": "frontend",
                "duration": 10_000,
                "attr.status_code": "OK",
            },
            {
                "time": ts1,
                "trace_id": "trace-1",
                "span_id": "child",
                "parent_span_id": "parent",
                "span_name": "POST /pay",
                "service_name": "payments",
                "duration": 25_000,
                "attr.status_code": "ERROR",
            },
        ],
    )
    _write_table(
        tmp_path / "normal_traces.parquet",
        [
            {
                "time": ts0,
                "trace_id": "trace-2",
                "span_id": "baseline-parent",
                "parent_span_id": None,
                "span_name": "GET /checkout",
                "service_name": "frontend",
                "duration": 9_000,
                "attr.status_code": "OK",
            }
        ],
    )
    _write_table(
        tmp_path / "abnormal_logs.parquet",
        [
            {
                "time": ts0,
                "level": "ERROR",
                "service_name": "checkout",
                "message": "checkout failed",
                "trace_id": "trace-1",
                "span_id": "child",
            }
        ],
    )
    _write_table(
        tmp_path / "normal_logs.parquet",
        [
            {
                "time": ts0,
                "level": "INFO",
                "service_name": "checkout",
                "message": "checkout steady",
                "trace_id": "trace-2",
                "span_id": "baseline-parent",
            }
        ],
    )

    try:
        yield tmp_path
    finally:
        close_obs_connection()
