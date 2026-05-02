from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from agentm.core._internal.catalog import _layout
from agentm.core._internal.catalog.indexer import index_trace, rebuild_catalog
from agentm.core.abi import EventBus
from agentm.harness.atom_reloader import AtomReloader
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_writer import GitBackedResourceWriter
from agentm.harness.session import AgentSession
from agentm.harness.session_manager import InMemorySessionManager


SHA_TOOL_LS = "a" * 40
SHA_OBS = "b" * 40
SHA_FIND = "c" * 40
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


def _fingerprint_record(atom_versions: dict[str, str]) -> dict[str, Any]:
    return _record(
        "session.fingerprint",
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
            _fingerprint_record({"tool_ls": SHA_TOOL_LS, "tool_find": SHA_FIND}),
            _record(
                "llm.request.end",
                {"usage": {"input_tokens": 120, "output_tokens": 30}},
            ),
            _record("agent_end", {"stop_reason": "end_turn"}),
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
                {"tool_ls": SHA_TOOL_LS, "tool_find": SHA_FIND, "observability": SHA_OBS}
            ),
            _record("agent_end", {"stop_reason": "stop"}),
        ],
    )

    result = index_trace(trace_path, root=tmp_path)

    assert result.n_atoms_attributed == 3
    for atom_name, version_hash in {
        "tool_ls": SHA_TOOL_LS,
        "tool_find": SHA_FIND,
        "observability": SHA_OBS,
    }.items():
        metrics_path = _layout.atom_metrics_path(atom_name, version_hash, root=tmp_path)
        assert metrics_path.is_file()


def test_index_trace_marks_mid_session_reload(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-reload",
        [
            _fingerprint_record({"tool_ls": SHA_TOOL_LS}),
            _record(
                "atom.reload",
                {
                    "fingerprint_after": {
                        "core": None,
                        "scenario": None,
                        "atoms": {"tool_ls": f"tool_ls@{SHA_TOOL_LS}"},
                    }
                },
            ),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )

    index_trace(trace_path, root=tmp_path)

    row = _first_metrics_row(tmp_path, "tool_ls", SHA_TOOL_LS)
    assert row["mid_session_reload"] is True


def test_index_trace_creates_runs_symlink_without_source_or_manifest(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-link",
        [
            _fingerprint_record({"tool_ls": SHA_TOOL_LS}),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )

    index_trace(trace_path, root=tmp_path)

    version_dir = _layout.atom_version_dir("tool_ls", SHA_TOOL_LS, root=tmp_path)
    assert version_dir.is_dir()
    assert not (version_dir / "source.py").exists()
    assert not (version_dir / "manifest.yaml").exists()
    runs_dir = _layout.atom_runs_dir("tool_ls", SHA_TOOL_LS, root=tmp_path)
    children = list(runs_dir.iterdir())
    assert [child.name for child in children] == ["trace-link"]
    assert children[0].is_symlink()


def test_index_trace_skips_legacy_content_hash_fingerprints(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "trace-legacy",
        [
            _fingerprint_record({"tool_ls": LEGACY_HASH}),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )

    result = index_trace(trace_path, root=tmp_path)

    assert result.n_atoms_attributed == 0
    assert result.warnings == [
        f"atom 'tool_ls' uses pre-migration fingerprint {LEGACY_HASH}; skipping"
    ]


def test_cli_rebuild_returns_zero_on_clean_run(tmp_path: Path) -> None:
    _write_trace(
        tmp_path,
        "trace-cli",
        [
            _fingerprint_record({"tool_ls": SHA_TOOL_LS}),
            _record("agent_end", {"stop_reason": "end_turn"}),
        ],
    )
    observability = tmp_path / ".agentm" / "observability"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm.core._internal.catalog.indexer",
            "rebuild",
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

    monkeypatch.setattr("agentm.core._internal.catalog.indexer.index_trace", _fake_index_trace)

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
    session = AgentSession(
        cwd=str(tmp_path),
        bus=bus,
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
        reloader=reloader,
        pending_user_messages=[],
        session_id="session-123",
        parent_bus=None,
        parent_session_id=None,
        purpose="test",
    )

    await session.shutdown()

    assert called["path"] == trace_path.resolve()
