"""Chain replay iteration smoke.

Mocks the underlying phase runner so the test stays hermetic — we only
want to check that ``chain_replay`` honors the phase filter, preserves
record order, and threads prompt overrides through to the right phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmharness.audit.runner import (
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    HarnessRunner,
)
from llmharness.audit.runner import runner as runner_module
from llmharness.audit.seams.offline import NoopSink
from llmharness.audit.toolkit.atom_constants import (
    EXTRACTOR_STATE_SERVICE_KEY,
    EXTRACTOR_TOOLS_MODULE,
)
from llmharness.replay import chain as chain_module
from llmharness.replay.record import ReplayRecord, write_record
from llmharness.tools.engine import PhaseResult

EVENT_1 = {
    "id": 1,
    "kind": "task",
    "summary": "investigate auth latency",
    "source_turns": [0],
    "external_refs": [],
}
EVENT_2 = {
    "id": 2,
    "kind": "act",
    "summary": "query auth spans for latency",
    "source_turns": [1],
    "external_refs": [],
}
EDGE_2_TO_1 = {
    "src": 2,
    "dst": 1,
    "kind": "data",
    "reason": "executes the auth latency investigation task",
    "src_turns": [1],
    "dst_turns": [0],
    "cited_entities": ["auth"],
    "cited_quote": "",
}
EVENT_1_OP = {
    "op": "node_upsert",
    "id": 1,
    "kind": "task",
    "summary": "investigate auth latency",
    "source_turns": [0],
    "external_refs": [],
}
EVENT_2_OP = {
    "op": "node_upsert",
    "id": 2,
    "kind": "act",
    "summary": "query auth spans for latency",
    "source_turns": [1],
    "external_refs": [],
}
EDGE_2_TO_1_OP = {
    "op": "edge_upsert",
    **EDGE_2_TO_1,
}


def _rec(phase: str, turn: int) -> ReplayRecord:
    return ReplayRecord(
        phase=phase,  # type: ignore[arg-type]
        turn_index=turn,
        session_id="sess-1",
        trace_id="trace-1",
        ts_ns=0,
        compose_kwargs={},
        payload={},
        provider=None,
        output={"recorded": True},
        status="ok",
        latency_ms=0,
    )


def _seed_sidecar(tmp_path: Path) -> Path:
    path = tmp_path / "replay.jsonl"
    write_record(path, _rec("extractor", 0))
    write_record(path, _rec("extractor", 1))
    write_record(path, _rec("auditor", 3))
    write_record(path, _rec("extractor", 2))
    write_record(path, _rec("auditor", 6))
    return path


def test_chain_replays_in_record_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int]] = []

    async def fake_extractor(record: ReplayRecord, **_: Any) -> PhaseResult:
        seen.append((record.phase, record.turn_index))
        return PhaseResult(output={"replayed": True}, status="ok", error=None, latency_ms=1, messages=[])

    async def fake_auditor(record: ReplayRecord, **_: Any) -> PhaseResult:
        seen.append((record.phase, record.turn_index))
        return PhaseResult(output={"replayed": True}, status="ok", error=None, latency_ms=2, messages=[])

    monkeypatch.setattr(chain_module, "replay_extractor_record", fake_extractor)
    monkeypatch.setattr(chain_module, "replay_auditor_record", fake_auditor)

    results = chain_module.chain_replay_sync(_seed_sidecar(tmp_path), cwd=str(tmp_path))
    assert [(r.record.phase, r.record.turn_index) for r in results] == [
        ("extractor", 0),
        ("extractor", 1),
        ("auditor", 3),
        ("extractor", 2),
        ("auditor", 6),
    ]
    assert seen == [(r.record.phase, r.record.turn_index) for r in results]


def test_chain_phase_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"extractor": 0, "auditor": 0}

    async def fake_extractor(record: ReplayRecord, **_: Any) -> PhaseResult:
        calls["extractor"] += 1
        return PhaseResult(output={}, status="ok", error=None, latency_ms=0, messages=[])

    async def fake_auditor(record: ReplayRecord, **_: Any) -> PhaseResult:
        calls["auditor"] += 1
        return PhaseResult(output={}, status="ok", error=None, latency_ms=0, messages=[])

    monkeypatch.setattr(chain_module, "replay_extractor_record", fake_extractor)
    monkeypatch.setattr(chain_module, "replay_auditor_record", fake_auditor)

    chain_module.chain_replay_sync(
        _seed_sidecar(tmp_path), cwd=str(tmp_path), phase="auditor"
    )
    # Three extractor records + two auditor records; phase=auditor must
    # skip the extractors entirely — that's the whole point of the filter.
    assert calls == {"extractor": 0, "auditor": 2}


def test_chain_threads_recent_edges_into_next_extractor_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "replay.jsonl"
    write_record(path, _rec("extractor", 0))
    write_record(path, _rec("extractor", 1))

    payloads: list[dict[str, Any]] = []

    async def fake_extractor(record: ReplayRecord, **_: Any) -> PhaseResult:
        payloads.append(dict(record.payload))
        if record.turn_index == 0:
            return PhaseResult(
                output={
                    "events": [EVENT_1, EVENT_2],
                    "edges": [EDGE_2_TO_1],
                    "dropped_edges": [],
                    "ops": [EVENT_1_OP, EVENT_2_OP, EDGE_2_TO_1_OP],
                },
                status="ok",
                error=None,
                latency_ms=1,
                messages=[],
            )
        return PhaseResult(
            output={"events": [], "edges": [], "dropped_edges": []},
            status="ok",
            error=None,
            latency_ms=1,
            messages=[],
        )

    monkeypatch.setattr(chain_module, "replay_extractor_record", fake_extractor)

    chain_module.chain_replay_sync(path, cwd=str(tmp_path), phase="extractor")

    assert payloads[0]["recent_graph"] == []
    assert payloads[0]["recent_edges"] == []
    assert payloads[0]["graph"] == {"nodes": [], "edges": []}
    assert payloads[1]["recent_graph"] == [EVENT_1, EVENT_2]
    assert payloads[1]["recent_edges"] == [EDGE_2_TO_1]
    assert payloads[1]["graph"] == {
        "nodes": [EVENT_1, EVENT_2],
        "edges": [EDGE_2_TO_1],
    }
    assert payloads[1]["next_event_id"] == 3


def test_chain_applies_extractor_ops_for_pure_historical_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "replay.jsonl"
    write_record(path, _rec("extractor", 0))
    write_record(path, _rec("extractor", 1))
    write_record(path, _rec("extractor", 2))

    payloads: list[dict[str, Any]] = []

    async def fake_extractor(record: ReplayRecord, **_: Any) -> PhaseResult:
        payloads.append(dict(record.payload))
        if record.turn_index == 0:
            return PhaseResult(
                output={
                    "events": [EVENT_1, EVENT_2],
                    "edges": [EDGE_2_TO_1],
                    "dropped_edges": [],
                    "ops": [EVENT_1_OP, EVENT_2_OP, EDGE_2_TO_1_OP],
                },
                status="ok",
                error=None,
                latency_ms=1,
                messages=[],
            )
        if record.turn_index == 1:
            return PhaseResult(
                output={
                    "events": [],
                    "edges": [],
                    "dropped_edges": [],
                    "ops": [{"op": "edge_delete", "src": 2, "dst": 1, "kind": "data"}],
                },
                status="ok",
                error=None,
                latency_ms=1,
                messages=[],
            )
        return PhaseResult(
            output={"events": [], "edges": [], "dropped_edges": [], "ops": []},
            status="ok",
            error=None,
            latency_ms=1,
            messages=[],
        )

    monkeypatch.setattr(chain_module, "replay_extractor_record", fake_extractor)

    chain_module.chain_replay_sync(path, cwd=str(tmp_path), phase="extractor")

    assert payloads[1]["graph"]["edges"] == [EDGE_2_TO_1]
    assert payloads[2]["graph"]["nodes"] == [EVENT_1, EVENT_2]
    assert payloads[2]["graph"]["edges"] == []


def test_extractor_record_replay_hydrates_recent_edges_into_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_phase_standalone(**kwargs: Any) -> PhaseResult:
        for module, cfg in kwargs["extensions"]:
            if module == EXTRACTOR_TOOLS_MODULE:
                captured["state"] = cfg[EXTRACTOR_STATE_SERVICE_KEY]
        captured["payload"] = kwargs["payload"]
        return PhaseResult(
            output=None,
            status="no_call",
            error="stop before model call",
            latency_ms=0,
            messages=[],
        )

    monkeypatch.setattr(
        runner_module, "run_phase_standalone", fake_run_phase_standalone
    )

    runner = HarnessRunner(
        cumulative=CumulativeAuditState.fresh(),
        child=None,  # type: ignore[arg-type]
        sink=NoopSink(),
        sidecar=None,
        extractor_settings=ExtractorSettings.from_compose_kwargs(
            {}, prompt_override="test prompt"
        ),
        auditor_settings=AuditorSettings.empty(),
        extractor_interval=1,
        audit_interval=1,
        enable_auditor=False,
        session_id="sess-1",
        trace_id="trace-1",
        provider_extractor=None,
        provider_auditor=None,
        cwd=str(tmp_path),
    )
    record = ReplayRecord(
        phase="extractor",
        turn_index=1,
        session_id="sess-1",
        trace_id="trace-1",
        ts_ns=0,
        compose_kwargs={},
        payload={
            "next_event_id": 3,
            "new_turns": [],
            "graph": {"nodes": [EVENT_1, EVENT_2], "edges": [EDGE_2_TO_1]},
        },
        provider=None,
        output=None,
        status="ok",
        latency_ms=0,
        extras={"turn_texts": {"0": "investigate auth latency", "1": "auth spans"}},
    )

    import asyncio

    asyncio.run(runner.fire_extractor_from_record(record))

    state = captured["state"]
    assert set(state.recent_graph_dict) == {1, 2}
    assert set(state.recent_edges_dict) == {(2, 1, "data")}
    assert set(state.pending_graph.nodes) == {1, 2}
    assert set(state.pending_graph.edges) == {(2, 1, "data")}
    assert captured["payload"]["graph"] == {
        "nodes": [
            {**EVENT_1, "source_turn_texts": ["investigate auth latency"]},
            {**EVENT_2, "source_turn_texts": ["auth spans"]},
        ],
        "edges": [EDGE_2_TO_1],
    }
    assert captured["payload"]["recent_graph"] == [
        {**EVENT_1, "source_turn_texts": ["investigate auth latency"]},
        {**EVENT_2, "source_turn_texts": ["auth spans"]},
    ]
    assert captured["payload"]["recent_edges"] == [EDGE_2_TO_1]
