"""Tests for the rca:baseline grader.

The grader scores a tuner trace by reading `.agentm/observability/<trace>.jsonl`
and matching the agent's `submit_final_report` against the expected service +
fault_kind. If the score function is wrong, the entire per-task-evolution
feedback loop trains on the wrong signal — making this an RCA-scenario
fail-stop.

Tests drive `grade()` with synthetic JSONL traces that mimic the emit shape
(session.fingerprint -> tool_call(list_tables) -> [optional Binder Error
tool_result] -> tool_call(submit_final_report)) and assert on the resulting
score plus the per-module credit-assignment payload.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_grader() -> Any:
    grader_path = (
        _REPO_ROOT
        / "contrib"
        / "scenarios"
        / "rca"
        / "eval"
        / "baseline"
        / "grader.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_test_grader_{uuid.uuid4().hex[:6]}", grader_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.grade


def _write_synthetic_trace(
    obs_dir: Path,
    *,
    task_id: str,
    service: str,
    fault_kind: str,
    include_binder_error: bool,
) -> None:
    obs_dir.mkdir(parents=True, exist_ok=True)
    trace_id = uuid.uuid4().hex
    path = obs_dir / f"{trace_id}.jsonl"
    now_ns = int(time.time() * 1e9)
    records: list[dict[str, Any]] = [
        {
            "schema": "otel/span/v0",
            "kind": "session.fingerprint",
            "trace_id": trace_id,
            "start_time_unix_nano": now_ns,
            "attributes": {
                "task_meta": {
                    "task_class": "rca_baseline",
                    "task_id": task_id,
                    "eval_run_id": "er_test",
                },
                "atoms": {},
            },
        },
        {
            "kind": "event.dispatch",
            "name": "emit:tool_call",
            "trace_id": trace_id,
            "attributes": {
                "channel": "tool_call",
                "event": {"tool_name": "list_tables", "args": {}},
            },
        },
    ]
    if include_binder_error:
        records.append(
            {
                "kind": "event.dispatch",
                "name": "emit:tool_result",
                "trace_id": trace_id,
                "attributes": {
                    "channel": "tool_result",
                    "event": {
                        "tool_name": "query_sql",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "error": (
                                                "query failed: Binder Error: "
                                                'Referenced table \\"attr\\" not found!'
                                            ),
                                            "sql": "SELECT attr.http.response.status_code FROM abnormal_traces",
                                        }
                                    ),
                                }
                            ],
                            "is_error": True,
                        },
                    },
                },
            }
        )
    records.append(
        {
            "kind": "event.dispatch",
            "name": "emit:tool_call",
            "trace_id": trace_id,
            "attributes": {
                "channel": "tool_call",
                "event": {
                    "tool_name": "submit_final_report",
                    "args": {
                        "root_causes": [
                            {
                                "service": service,
                                "fault_kind": fault_kind,
                                "evidence": [
                                    {
                                        "kind": "metric",
                                        "sql": "SELECT 1",
                                        "claim": "synthetic",
                                    }
                                ],
                            }
                        ]
                    },
                },
            },
        }
    )
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_grader_scores_correct_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_synthetic_trace(
        tmp_path / ".agentm" / "observability",
        task_id="01_mysql_corrupt",
        service="ts-station-service",
        fault_kind="network_corrupt",
        include_binder_error=True,
    )
    grade = _load_grader()
    task = {
        "id": "01_mysql_corrupt",
        "expected": {
            "expected_services": ["mysql", "ts-station-service"],
            "fault_kind": "network_corrupt",
        },
    }
    result = grade(task, "ignored")
    assert result["score"] == pytest.approx(1.0)
    assert "query_sql" in result["module_feedback"]


def test_grader_scores_wrong_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_synthetic_trace(
        tmp_path / ".agentm" / "observability",
        task_id="01_mysql_corrupt",
        service="ts-foo-service",
        fault_kind="cpu_stress",
        include_binder_error=False,
    )
    grade = _load_grader()
    task = {
        "id": "01_mysql_corrupt",
        "expected": {
            "expected_services": ["mysql", "ts-station-service"],
            "fault_kind": "network_corrupt",
        },
    }
    result = grade(task, "ignored")
    assert result["score"] == pytest.approx(0.0)
    assert "query_sql" not in result["module_feedback"]
