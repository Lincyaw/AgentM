"""Chain replay iteration smoke.

Mocks the underlying phase runner so the test stays hermetic — we only
want to check that ``chain_replay`` honors the phase filter, preserves
record order, and threads prompt overrides through to the right phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmharness.replay import chain as chain_module
from llmharness.replay.record import ReplayRecord, write_record
from llmharness.tools.engine import PhaseResult


def _rec(phase: str, turn: int) -> ReplayRecord:
    return ReplayRecord(
        phase=phase,  # type: ignore[arg-type]
        turn_index=turn,
        root_session_id="trace-1",
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


def test_chain_threads_phase_specific_prompt_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_prompts: dict[str, str | None] = {"extractor": None, "auditor": None}

    async def fake_extractor(
        record: ReplayRecord, *, cwd: str, provider_override: Any, prompt_override: str | None
    ) -> PhaseResult:
        seen_prompts["extractor"] = prompt_override
        return PhaseResult(output={}, status="ok", error=None, latency_ms=0, messages=[])

    async def fake_auditor(
        record: ReplayRecord, *, cwd: str, provider_override: Any, prompt_override: str | None
    ) -> PhaseResult:
        seen_prompts["auditor"] = prompt_override
        return PhaseResult(output={}, status="ok", error=None, latency_ms=0, messages=[])

    monkeypatch.setattr(chain_module, "replay_extractor_record", fake_extractor)
    monkeypatch.setattr(chain_module, "replay_auditor_record", fake_auditor)

    chain_module.chain_replay_sync(
        _seed_sidecar(tmp_path),
        cwd=str(tmp_path),
        prompt_override_extractor="EXT_PROMPT",
        prompt_override_auditor="AUD_PROMPT",
    )
    assert seen_prompts == {"extractor": "EXT_PROMPT", "auditor": "AUD_PROMPT"}
