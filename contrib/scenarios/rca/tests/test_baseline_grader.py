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


def _otlp_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"stringValue": ""}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(k), "value": _otlp_value(v)}
                    for k, v in value.items()
                ]
            }
        }
    if isinstance(value, list):
        return {"arrayValue": {"values": [_otlp_value(v) for v in value]}}
    return {"stringValue": json.dumps(value, default=str)}


def _otlp_log(event_name: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "agentm"}}
            ]
        },
        "scopeLogs": [
            {
                "scope": {"name": "agentm", "version": "0.1.0"},
                "logRecords": [
                    {
                        "timeUnixNano": "0",
                        "observedTimeUnixNano": "0",
                        "severityNumber": "SEVERITY_NUMBER_INFO",
                        "severityText": "INFO",
                        "eventName": event_name,
                        "body": _otlp_value(body),
                    }
                ],
            }
        ],
    }


def _otlp_tool_span(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "agentm"}}
            ]
        },
        "scopeSpans": [
            {
                "scope": {"name": "agentm", "version": "0.1.0"},
                "spans": [
                    {
                        "traceId": "AAAA",
                        "spanId": "AAAA",
                        "name": f"execute_tool {tool_name}",
                        "kind": "SPAN_KIND_INTERNAL",
                        "startTimeUnixNano": "0",
                        "endTimeUnixNano": "1",
                        "attributes": [
                            {
                                "key": "gen_ai.tool.name",
                                "value": {"stringValue": tool_name},
                            },
                            {
                                "key": "gen_ai.tool.call.arguments",
                                "value": {"stringValue": json.dumps(args)},
                            },
                        ],
                        "status": {},
                    }
                ],
            }
        ],
    }


def _otlp_tool_result_log(
    tool_name: str, result_text: str
) -> dict[str, Any]:
    """Emit an ``agentm.tool.call.result`` log carrying ``result_text``.

    The grader's binder-error detection is a substring scan over the raw
    trace file (``_detect_sql_quoting_issue``); placing the marker text
    inside a tool-result log record gives the scan something to find
    without depending on any single span/log shape.
    """

    return {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "agentm"}}
            ]
        },
        "scopeLogs": [
            {
                "scope": {"name": "agentm", "version": "0.1.0"},
                "logRecords": [
                    {
                        "timeUnixNano": "0",
                        "observedTimeUnixNano": "0",
                        "severityNumber": "SEVERITY_NUMBER_INFO",
                        "severityText": "INFO",
                        "eventName": "agentm.tool.call.result",
                        "body": _otlp_value(
                            {"tool": tool_name, "text": result_text}
                        ),
                    }
                ],
            }
        ],
    }


def _write_synthetic_trace(
    obs_dir: Path,
    *,
    task_id: str,
    service: str,
    fault_kind: str,
    include_binder_error: bool,
) -> None:
    """Write an OTLP/JSON trace with fingerprint identity + a
    submit_final_report verdict span. The grader (now OTLP-aware) reads
    the fingerprint from log records and the verdict args from the
    execute_tool span. When ``include_binder_error`` is set the fixture
    also emits a query_sql tool-result log whose text contains the
    canonical DuckDB ``Binder Error / Referenced table / attr`` triple
    so ``_detect_sql_quoting_issue`` finds the SQL quoting hint.
    """
    obs_dir.mkdir(parents=True, exist_ok=True)
    trace_id = uuid.uuid4().hex
    path = obs_dir / f"{trace_id}.jsonl"
    _ = (time, trace_id)
    records: list[dict[str, Any]] = [
        _otlp_log(
            "agentm.session.fingerprint",
            {
                "task_meta": {
                    "task_class": "rca_baseline",
                    "task_id": task_id,
                    "eval_run_id": "er_test",
                },
                "atoms": {},
            },
        ),
    ]
    if include_binder_error:
        records.append(
            _otlp_tool_result_log(
                "query_sql",
                f"task_id={task_id} Binder Error: Referenced table "
                f'"attr" was not found in any catalog',
            )
        )
    records.append(
        _otlp_tool_span(
            "submit_final_report",
            {
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
        ),
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
