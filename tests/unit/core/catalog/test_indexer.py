from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from agentm.core.catalog import _layout
from agentm.core.catalog.hashing import compute_atom_hash
from agentm.core.catalog.indexer import index_trace, rebuild_catalog
from agentm.core.kernel import EventBus
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession
from agentm.harness.session_manager import InMemorySessionManager
from agentm.extensions.discover import discover_builtin


BUILTIN_ATOMS = ("tool_ls", "tool_find", "observability")


def _builtin_hashes(*names: str) -> dict[str, str]:
    catalog = discover_builtin()
    hashes: dict[str, str] = {}
    for name in names:
        entry = catalog[name]
        source_path = inspect.getsourcefile(entry.module)
        assert source_path is not None
        source = Path(source_path).read_text(encoding="utf-8")
        hashes[name] = compute_atom_hash(source)
    return hashes


def _write_trace(tmp_path: Path, trace_id: str, records: list[dict[str, Any]]) -> Path:
    observability_dir = tmp_path / ".agentm" / "observability"
    observability_dir.mkdir(parents=True, exist_ok=True)
    trace_path = observability_dir / f"{trace_id}.jsonl"
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return trace_path


def _record(kind: str, attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "otel/span/v0",
        "kind": kind,
        "trace_id": "trace",
        "span_id": "span",
        "name": kind,
        "attributes": attributes,
        "status": {"code": "OK"},
    }


def _fingerprint_record(atom_hashes: dict[str, str]) -> dict[str, Any]:
    return _record(
        "session.fingerprint",
        {
            "core": None,
            "scenario": None,
            "atoms": {
                name: f"{name}@{version_hash}"
                for name, version_hash in atom_hashes.items()
            },
        },
    )


def _strip_indexed_at(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    copied.pop("indexed_at", None)
    return copied


def _metrics_snapshot(root: Path) -> dict[str, list[dict[str, Any]]]:
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for metrics_path in sorted(root.glob("atoms/*/*/metrics.jsonl")):
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
    atom_hashes = _builtin_hashes(*BUILTIN_ATOMS)
    trace_path = _write_trace(
        tmp_path,
        "trace-e5",
        [
            _fingerprint_record(atom_hashes),
            _record(
                "llm.request.end",
                {"usage": {"input_tokens": 120, "output_tokens": 30}},
            ),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    root = tmp_path

    index_trace(trace_path, root=root)
    original = _metrics_snapshot(root)

    n_traces, n_atoms, n_warnings, failures = rebuild_catalog(
        root=root,
        observability=tmp_path / ".agentm" / "observability",
    )

    assert (n_traces, n_atoms, n_warnings, failures) == (1, 3, 0, 0)
    assert _metrics_snapshot(root) == original


def test_index_trace_attributes_to_all_loaded_atoms(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes(*BUILTIN_ATOMS)
    trace_path = _write_trace(
        tmp_path,
        "trace-atoms",
        [
            _fingerprint_record(atom_hashes),
            _record("agent_end", {"stop_reason": "stop"}),
        ],
    )
    root = tmp_path

    result = index_trace(trace_path, root=root)

    assert result.n_atoms_attributed == 3
    for atom_name, version_hash in atom_hashes.items():
        metrics_path = _layout.atom_metrics_path(atom_name, version_hash, root=root)
        assert metrics_path.is_file()


def test_index_trace_marks_mid_session_reload(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes("tool_ls")
    trace_path = _write_trace(
        tmp_path,
        "trace-reload",
        [
            _fingerprint_record(atom_hashes),
            _record(
                "atom.reload",
                {
                    "fingerprint_after": {
                        "core": None,
                        "scenario": None,
                        "atoms": {"tool_ls": f"tool_ls@{atom_hashes['tool_ls']}"},
                    }
                },
            ),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    root = tmp_path

    index_trace(trace_path, root=root)

    row = _first_metrics_row(root, "tool_ls", atom_hashes["tool_ls"])
    assert row["mid_session_reload"] is True


def test_index_trace_lazily_freezes_genesis_version(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes("tool_ls")
    trace_path = _write_trace(
        tmp_path,
        "trace-freeze",
        [
            _fingerprint_record(atom_hashes),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    root = tmp_path

    index_trace(trace_path, root=root)

    version_dir = _layout.atom_version_dir("tool_ls", atom_hashes["tool_ls"], root=root)
    assert version_dir.is_dir()
    assert (version_dir / "source.py").is_file()
    assert (version_dir / "manifest.yaml").is_file()
    assert (version_dir / "runs").is_dir()


def test_index_trace_handles_missing_fingerprint_record(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-missing-fingerprint",
        [_record("session.start", {"cwd": str(tmp_path)})],
    )

    result = index_trace(trace_path, root=tmp_path / ".agentm" / "catalog")

    assert result.n_atoms_attributed == 0
    assert result.warnings
    assert "missing session.fingerprint" in result.warnings[0]


def test_completion_rate_one_for_end_turn(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes("tool_ls")
    trace_path = _write_trace(
        tmp_path,
        "trace-end-turn",
        [
            _fingerprint_record(atom_hashes),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    root = tmp_path

    index_trace(trace_path, root=root)

    row = _first_metrics_row(root, "tool_ls", atom_hashes["tool_ls"])
    assert row["metrics"]["task.completion_rate"] == 1.0


def test_completion_rate_zero_for_budget_stop(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes("tool_ls")
    trace_path = _write_trace(
        tmp_path,
        "trace-budget",
        [
            _fingerprint_record(atom_hashes),
            _record("agent_end", {"stop_reason": "budget"}),
        ],
    )
    root = tmp_path

    index_trace(trace_path, root=root)

    row = _first_metrics_row(root, "tool_ls", atom_hashes["tool_ls"])
    assert row["metrics"]["task.completion_rate"] == 0.0


def test_runs_symlink_created_idempotently(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes("tool_ls")
    trace_path = _write_trace(
        tmp_path,
        "trace-link",
        [
            _fingerprint_record(atom_hashes),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    root = tmp_path

    index_trace(trace_path, root=root)
    index_trace(trace_path, root=root)

    runs_dir = _layout.atom_runs_dir("tool_ls", atom_hashes["tool_ls"], root=root)
    children = list(runs_dir.iterdir())
    assert [child.name for child in children] == ["trace-link"]
    assert children[0].is_symlink()


def test_cli_rebuild_returns_zero_on_clean_run(tmp_path: Path) -> None:
    atom_hashes = _builtin_hashes("tool_ls")
    _write_trace(
        tmp_path,
        "trace-cli",
        [
            _fingerprint_record(atom_hashes),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    root = tmp_path
    observability = tmp_path / ".agentm" / "observability"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm.core.catalog.indexer",
            "rebuild",
            "--root",
            str(root),
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

    monkeypatch.setattr("agentm.core.catalog.indexer.index_trace", _fake_index_trace)

    session = AgentSession(
        cwd=str(tmp_path),
        bus=EventBus(),
        session_manager=InMemorySessionManager(cwd=str(tmp_path)),
        resource_loader=None,  # type: ignore[arg-type]
        loop=None,  # type: ignore[arg-type]
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
        command_owners={},
        loaded_atoms_by_name={},
        pending_user_messages=[],
        session_id="session-123",
        parent_bus=None,
        parent_session_id=None,
        purpose="test",
    )

    await session.shutdown()

    assert called["path"] == trace_path.resolve()


