from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from agentm.core.runtime.catalog import _layout
from agentm.core.runtime.catalog.indexer import index_trace, rebuild_catalog
from agentm.core.abi import EventBus
from agentm.core.runtime.atom_reloader import AtomReloader
from agentm.core.abi.extension import ProviderConfig
from agentm.core.runtime.resource_writer import GitBackedResourceWriter
from agentm.core.runtime.session import AgentSession
from agentm.core.runtime.session_manager import InMemorySessionManager
from agentm.core.runtime.session_runtime import SessionRuntime


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
                {
                    "key": "agentm.session.id",
                    "value": {"stringValue": "sess-fixture"},
                },
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


def test_index_trace_marks_mid_session_reload(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-reload",
        [
            _fingerprint_record({"tool_read": SHA_TOOL_READ}),
            _record(
                "agentm.atom.reload",
                {
                    "fingerprint_after": {
                        "core": None,
                        "scenario": None,
                        "atoms": {"tool_read": f"tool_read@{SHA_TOOL_READ}"},
                    }
                },
            ),
            _record("agentm.agent.end", {"stop_reason": "end_turn"}),
        ],
    )

    index_trace(trace_path, root=tmp_path)

    row = _first_metrics_row(tmp_path, "tool_read", SHA_TOOL_READ)
    assert row["mid_session_reload"] is True


@pytest.mark.parametrize(
    ("cause_payload", "expected_completion_rate"),
    [
        # ModelEndTurn — model voluntarily finished.
        ({"cause_kind": "ModelEndTurn", "final": False}, 1.0),
        # ToolTerminated — terminal tool ran to completion (e.g. RCA's
        # submit_final_report).
        (
            {
                "cause_kind": "ToolTerminated",
                "final": False,
                "tool_name": "submit_final_report",
                "reason": "done",
            },
            1.0,
        ),
        # ProviderTruncated(kind=max_tokens) — fail-stop. ``cause_kind``
        # carries the class name; ``kind`` is the dataclass field.
        (
            {
                "cause_kind": "ProviderTruncated",
                "final": False,
                "kind": "max_tokens",
            },
            0.0,
        ),
        # ProviderTruncated(kind=error) — fail-stop.
        (
            {"cause_kind": "ProviderTruncated", "final": False, "kind": "error"},
            0.0,
        ),
        # ProviderProtocolViolation — fail-stop.
        (
            {
                "cause_kind": "ProviderProtocolViolation",
                "final": False,
                "detail": "tool_use without tool_calls",
            },
            0.0,
        ),
        # MaxTurnsExhausted — bare class with no fields.
        ({"cause_kind": "MaxTurnsExhausted", "final": True}, 0.0),
        # SignalAborted — bare class with no fields.
        ({"cause_kind": "SignalAborted", "final": True}, 0.0),
        # BudgetExhausted — ``final=True`` with discriminating ``detail``.
        (
            {"cause_kind": "BudgetExhausted", "final": True, "detail": "cost"},
            0.0,
        ),
    ],
)
def test_extract_stop_reason_handles_new_cause_shapes(
    tmp_path: Path,
    cause_payload: dict[str, Any],
    expected_completion_rate: float,
) -> None:
    """Migration coverage: each serialized ``TerminationCause`` shape must
    classify correctly. Without this guard a mixed-vintage trace store
    (legacy ``stop_reason`` strings + new ``cause`` dicts) silently
    miscounts ``completion_rate`` while CI stays green — and
    completion_rate is the key evidence-driven-evolution signal."""

    trace_path = _write_trace(
        tmp_path,
        "trace-cause",
        [
            _fingerprint_record({"tool_read": SHA_TOOL_READ}),
            _record("agentm.agent.end", {"cause": cause_payload}),
        ],
    )

    index_trace(trace_path, root=tmp_path)
    row = _first_metrics_row(tmp_path, "tool_read", SHA_TOOL_READ)
    assert row["metrics"]["task.completion_rate"] == expected_completion_rate


def test_index_trace_skips_legacy_content_hash_fingerprints(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-legacy",
        [
            _fingerprint_record({"tool_read": LEGACY_HASH}),
            _record("agentm.agent.end", {"stop_reason": "end_turn"}),
        ],
    )

    result = index_trace(trace_path, root=tmp_path)

    assert result.n_atoms_attributed == 0
    assert result.warnings == [
        f"atom 'tool_read' uses pre-migration fingerprint {LEGACY_HASH}; skipping"
    ]


def test_cli_rebuild_returns_zero_on_clean_run(tmp_path: Path) -> None:
    _write_trace(
        tmp_path,
        "trace-cli",
        [
            _fingerprint_record({"tool_read": SHA_TOOL_READ}),
            _record("agentm.agent.end", {"stop_reason": "end_turn"}),
        ],
    )
    observability = tmp_path / ".agentm" / "observability"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm.core.runtime.catalog.indexer",
            "--root",
            str(tmp_path),
            "--observability",
            str(observability),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0
    assert "n_traces=1" in completed.stdout


@pytest.mark.asyncio
async def test_shutdown_indexes_observability_trace_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = tmp_path / ".agentm" / "observability"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "session-123.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")

    called: dict[str, Path] = {}

    def _fake_index_trace(path: Path, *, root: Path | None = None) -> None:
        called["path"] = path

    monkeypatch.setattr("agentm.core.runtime.catalog.indexer.index_trace", _fake_index_trace)

    bus = EventBus()
    resource_writer = GitBackedResourceWriter(
        cwd=str(tmp_path),
        session_id="session-123",
        bus=bus,
    )
    reloader = AtomReloader(
        cwd=str(tmp_path),
        resource_writer=resource_writer,
        bus=bus,
        tools=[],
        commands={},
        providers={},
        renderers={},
        apis={},
        on_provider_changed=lambda: None,
    )
    runtime = SessionRuntime(
        bus=bus,
        session_manager=InMemorySessionManager(cwd=str(tmp_path)),
        resource_loader=cast("Any", None),
        loop=cast("Any", None),
        active_provider_box={
            "value": ProviderConfig(
                stream_fn=cast("Any", lambda *_args, **_kwargs: None),
                model=cast("Any", None),
                name="dummy",
            )
        },
        tools=[],
        commands={},
        providers={},
        renderers={},
        apis={},
        services={},
        reloader=reloader,
        pending_user_messages=[],
    )
    session = AgentSession(
        cwd=str(tmp_path),
        runtime=runtime,
        session_id="session-123",
        parent_bus=None,
        parent_session_id=None,
    )

    await session.shutdown()

    assert called["path"] == trace_path.resolve()
