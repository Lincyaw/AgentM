from __future__ import annotations

from pathlib import Path

import pytest

from contrib.scenarios.rca.eval import grader


def _write_parquet(path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect(":memory:")
    try:
        target = path.as_posix().replace("'", "''")
        conn.execute(
            "COPY (SELECT 1 AS service_id, 42 AS latency_ms) "
            f"TO '{target}' (FORMAT PARQUET)"
        )
    finally:
        conn.close()


def test_fpg_sql_evidence_eval_replays_final_sql(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "abnormal_traces.parquet")
    verdict = {
        "fpg_output": {
            "nodes": [
                {
                    "id": "n1",
                    "subject": "svc:checkout",
                    "predicate": "latency_degraded",
                    "evidence": [
                        {
                            "query": {
                                "language": "sql",
                                "statement": (
                                    "SELECT p95(latency_ms) AS p95_latency "
                                    "FROM abnormal_traces"
                                ),
                            },
                            "explanation": "latency rose",
                        },
                        {
                            "query": {
                                "language": "sql",
                                "statement": (
                                    "SELECT missing_column FROM abnormal_traces"
                                ),
                            },
                            "explanation": "bad evidence",
                        },
                    ],
                }
            ],
            "edges": [],
            "root_causes": ["n1"],
        }
    }

    result = grader._evaluate_fpg_sql_evidence(
        {"meta": {"case_dir": str(tmp_path)}}, verdict
    )

    assert result is not None
    assert result["total"] == 2
    assert result["executable"] == 1
    assert result["failed"] == 1
    assert result["ratio"] == 0.5
    assert result["all_executable"] is False
    assert result["failures"][0]["node_id"] == "n1"
    assert result["failures"][0]["evidence_index"] == 1


def test_sql_evidence_dimensions_are_reported() -> None:
    dimensions = grader._sql_evidence_dimensions(
        {
            "total": 2,
            "executable": 1,
            "failed": 1,
            "ratio": 0.5,
            "all_executable": False,
        }
    )

    assert dimensions == {
        "fpg_sql_evidence_count": 2.0,
        "fpg_sql_executable_count": 1.0,
        "fpg_sql_failed_count": 1.0,
        "fpg_sql_executable_ratio": 0.5,
        "fpg_sql_all_executable": 0.0,
    }
