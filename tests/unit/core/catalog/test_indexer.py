from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from agentm.core.runtime.catalog import _layout
from agentm.core.runtime.catalog.indexer import index_trace, rebuild_catalog


SHA_TOOL_READ = "a" * 40
SHA_OBS = "b" * 40
SHA_TOOL_WRITE = "c" * 40
LEGACY_HASH = "deadbeef"


def _write_trace(tmp_path: Path, trace_id: str, records: list[dict[str, Any]]) -> Path:
    observability_dir = tmp_path / ".agentm" / "observability"
    observability_dir.mkdir(parents=True, exist_ok=True)
    trace_path = observability_dir / f"{trace_id}.jsonl"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return trace_path


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


def _record(event_name: str, body: dict[str, Any]) -> dict[str, Any]:
    """Build one ``ResourceLogs`` element line carrying one log record.

    The indexer walks ``scopeLogs[*].logRecords[*]`` and matches on
    ``eventName``; this mirrors the on-disk shape PR-A's
    ``FileLogExporter`` emits.
    """
    return {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "agentm"}},
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


def _fingerprint_record(atom_versions: dict[str, str]) -> dict[str, Any]:
    return _record(
        "agentm.session.fingerprint",
        {
            "core": None,
            "scenario": None,
            "atoms": {
                name: f"{name}@{version_hash}"
                for name, version_hash in atom_versions.items()
            },
        },
    )


def _strip_indexed_at(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    copied.pop("indexed_at", None)
    return copied


def _metrics_snapshot(root: Path) -> dict[str, list[dict[str, Any]]]:
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for metrics_path in sorted(root.glob(".agentm/catalog/atoms/*/*/metrics.jsonl")):
        snapshot[str(metrics_path.relative_to(root))] = [
            _strip_indexed_at(json.loads(line))
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
    return snapshot


def _first_metrics_row(root: Path, atom_name: str, version_hash: str) -> dict[str, Any]:
    metrics_path = _layout.atom_metrics_path(atom_name, version_hash, root=root)
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert lines
    return json.loads(lines[0])


def test_E5_rebuild_is_idempotent(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-e5",
        [
            _fingerprint_record({"tool_read": SHA_TOOL_READ, "tool_write": SHA_TOOL_WRITE}),
            _record(
                "agentm.turn.summary",
                {"input_tokens": 120, "output_tokens": 30},
            ),
            _record("agentm.agent.end", {"stop_reason": "end_turn"}),
        ],
    )

    index_trace(trace_path, root=tmp_path)
    original = _metrics_snapshot(tmp_path)

    n_traces, n_atoms, n_warnings, failures = rebuild_catalog(
        root=tmp_path,
        observability=tmp_path / ".agentm" / "observability",
    )

    assert (n_traces, n_atoms, n_warnings, failures) == (1, 2, 0, 0)
    assert _metrics_snapshot(tmp_path) == original


def test_index_trace_attributes_to_all_loaded_atoms(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-atoms",
        [
            _fingerprint_record(
                {
                    "tool_read": SHA_TOOL_READ,
                    "tool_write": SHA_TOOL_WRITE,
                    "observability": SHA_OBS,
                }
            ),
            _record("agentm.agent.end", {"stop_reason": "stop"}),
        ],
    )

    result = index_trace(trace_path, root=tmp_path)

    assert result.n_atoms_attributed == 3
    for atom_name, version_hash in {
        "tool_read": SHA_TOOL_READ,
        "tool_write": SHA_TOOL_WRITE,
        "observability": SHA_OBS,
    }.items():
        metrics_path = _layout.atom_metrics_path(atom_name, version_hash, root=tmp_path)
        assert metrics_path.is_file()










