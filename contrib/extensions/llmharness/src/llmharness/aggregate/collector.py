"""Replay sidecar + meta sidecar → :class:`CaseData`.

Pure function. No I/O writes; reads only the two sidecar files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..distill.binding import read_sample_meta
from ..replay.record import ReplayRecord, iter_records
from .case import CaseData, CaseMeta, FiringRecord, FiringStatus, GraphSnapshot


def _firing_input_payload(rec: ReplayRecord) -> dict[str, Any]:
    """Per-firing input view: the user-visible payload that was sent to
    the child, plus the resolved framing + structural inputs from
    ``compose_kwargs``. We drop the framing prompt itself (long, lives
    elsewhere) but keep its identity if present.
    """
    ck = rec.compose_kwargs or {}
    out: dict[str, Any] = {
        "payload": rec.payload,
        "summary_threshold": ck.get("summary_threshold"),
    }
    if rec.phase == "auditor":
        out["findings"] = ck.get("findings") or []
        out["check_errors"] = ck.get("check_errors") or {}
        out["continuation_notes"] = ck.get("continuation_notes") or []
        out["tools"] = ck.get("tools") or []
    return out


def _firing_status(rec: ReplayRecord) -> FiringStatus:
    raw = rec.status
    if raw == "ok":
        return "ok"
    if raw == "no_call":
        return "no_call"
    if raw == "spawn_error":
        return "spawn_error"
    if raw == "prompt_error":
        return "prompt_error"
    return "ok"


def _to_firing(rec: ReplayRecord, sequence: int) -> FiringRecord:
    return FiringRecord(
        phase=rec.phase,
        sequence=sequence,
        turn_index=int(rec.turn_index),
        ts_ns=int(rec.ts_ns or 0),
        input_payload=_firing_input_payload(rec),
        output=rec.output,
        status=_firing_status(rec),
        error=rec.error,
        latency_ms=int(rec.latency_ms or 0),
    )


def _accumulate_graph(
    extractor_firings: list[FiringRecord],
) -> list[GraphSnapshot]:
    """One snapshot per ok extractor firing, accumulating events + edges.

    Non-ok firings produce no snapshot — the cursor doesn't advance and
    nothing new lands in the graph.
    """
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    snapshots: list[GraphSnapshot] = []
    for fr in extractor_firings:
        if fr.status != "ok" or fr.output is None:
            continue
        new_events = fr.output.get("events") or []
        new_edges = fr.output.get("edges") or []
        events = [*events, *new_events]
        edges = [*edges, *new_edges]
        snapshots.append(
            GraphSnapshot(
                after_extractor_firing=fr.sequence,
                turn_index=fr.turn_index,
                events=events,
                edges=edges,
            )
        )
    return snapshots


def _verdicts_from_auditor(auditor_firings: list[FiringRecord]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fr in auditor_firings:
        if fr.status != "ok" or fr.output is None:
            continue
        row = {
            "sequence": fr.sequence,
            "turn_index": fr.turn_index,
            "ts_ns": fr.ts_ns,
        }
        row.update(fr.output)
        out.append(row)
    return out


def _main_agent_messages(auditor_firings_records: list[ReplayRecord]) -> list[dict[str, Any]]:
    """Pull the most complete trajectory snapshot.

    The auditor receives the full serialised trajectory as
    ``compose_kwargs.trajectory_snapshot`` at every firing — the last
    ok firing's snapshot has the most coverage. Falls back to any
    snapshot if no ok firings exist.
    """
    for rec in reversed(auditor_firings_records):
        if rec.status == "ok":
            snap = (rec.compose_kwargs or {}).get("trajectory_snapshot")
            if isinstance(snap, list):
                return list(snap)
    for rec in reversed(auditor_firings_records):
        snap = (rec.compose_kwargs or {}).get("trajectory_snapshot")
        if isinstance(snap, list):
            return list(snap)
    return []


def collect_case(
    *,
    replay_path: Path,
    meta_path: Path | None = None,
) -> CaseData:
    """Build a :class:`CaseData` from one replay sidecar.

    ``meta_path`` is optional; when missing, ``sample_id`` /
    ``dataset_*`` fields fall back to ``None`` and ``case_id`` derives
    from the replay file stem (the root_session_id).
    """
    records = list(iter_records(replay_path))
    extractor_records = [r for r in records if r.phase == "extractor"]
    auditor_records = [r for r in records if r.phase == "auditor"]

    extractor_firings = [
        _to_firing(r, seq) for seq, r in enumerate(extractor_records, start=1)
    ]
    auditor_firings = [
        _to_firing(r, seq) for seq, r in enumerate(auditor_records, start=1)
    ]

    meta_obj = read_sample_meta(meta_path) if meta_path is not None else None
    sample_id = meta_obj.sample_id if meta_obj is not None else None
    dataset_name = meta_obj.dataset_name if meta_obj is not None else None
    dataset_path = meta_obj.dataset_path if meta_obj is not None else None

    root_session_id = records[0].root_session_id if records else replay_path.stem
    case_id = sample_id or root_session_id

    ts_values = [r.ts_ns for r in records if r.ts_ns]
    started_at = min(ts_values) if ts_values else 0
    ended_at = max(ts_values) if ts_values else 0

    verdicts = _verdicts_from_auditor(auditor_firings)
    surfaced = sum(1 for v in verdicts if v.get("surface_reminder"))
    silent = sum(1 for v in verdicts if not v.get("surface_reminder"))

    meta = CaseMeta(
        case_id=case_id,
        root_session_id=root_session_id,
        sample_id=sample_id,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        started_at_ns=started_at,
        ended_at_ns=ended_at,
        extractor_firings=len(extractor_firings),
        auditor_firings=len(auditor_firings),
        surfaced_reminders=surfaced,
        silent_verdicts=silent,
    )

    return CaseData(
        meta=meta,
        main_agent_messages=_main_agent_messages(auditor_records),
        extractor_firings=extractor_firings,
        auditor_firings=auditor_firings,
        graph_snapshots=_accumulate_graph(extractor_firings),
        verdicts=verdicts,
    )


__all__ = ["collect_case"]
