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

    Each extractor call emits a *local* id space starting at 1 (events
    1..N and edges referencing those locals). Concatenating raw outputs
    would collide ids across firings, so renumber every event with a
    monotonic global id at the moment it enters the cumulative graph,
    and rewrite the firing's edges (and its cross-firing external_refs)
    to point at the same globals.

    Cross-firing connectivity arrives via ``events[].external_refs[]``:
    each ``to_recent_graph_index`` is a 1-based index into the
    ``recent_graph`` slice the live harness presented to that firing,
    which the live adapter selects as the most recent
    ``_RECENT_GRAPH_SLICE_FOR_EXTRACTOR`` events of the running graph.
    We rebuild that slice here against ``cum_events`` BEFORE this
    firing's events are merged, so resolution matches what the
    extractor actually saw.

    The per-firing JSON files on disk are NOT renumbered — they remain
    the raw extractor output. Only the cumulative snapshot is
    canonicalised; that's the contract documented in 02-schemas.md.
    """
    from ..audit.entry_types import RECENT_GRAPH_SLICE_FOR_EXTRACTOR

    _RECENT_SLICE = RECENT_GRAPH_SLICE_FOR_EXTRACTOR

    cum_events: list[dict[str, Any]] = []
    cum_edges: list[dict[str, Any]] = []
    snapshots: list[GraphSnapshot] = []
    next_id = 0
    for fr in extractor_firings:
        if fr.status != "ok" or fr.output is None:
            continue
        new_events = fr.output.get("events") or []
        new_edges = fr.output.get("edges") or []

        # The recent_graph slice the extractor saw is the tail of the
        # cumulative graph *before* this firing's contribution lands.
        recent_slice = cum_events[-_RECENT_SLICE:] if cum_events else []

        local_to_global: dict[int, int] = {}
        relabelled_events: list[dict[str, Any]] = []
        external_edges: list[dict[str, Any]] = []
        for ev in new_events:
            local_id = ev.get("id")
            if not isinstance(local_id, int):
                relabelled_events.append(ev)
                continue
            next_id += 1
            local_to_global[local_id] = next_id
            # Drop external_refs from the stored event — they only
            # carry information until we resolve them to edges; keeping
            # them on the snapshot event would be redundant and confuse
            # downstream consumers that scan event payloads for edges.
            event_copy = {k: v for k, v in ev.items() if k != "external_refs"}
            event_copy["id"] = next_id
            relabelled_events.append(event_copy)

            ext_refs = ev.get("external_refs") or []
            for ext in ext_refs:
                if not isinstance(ext, dict):
                    continue
                idx = ext.get("to_recent_graph_index")
                if not isinstance(idx, int) or idx < 1 or idx > len(recent_slice):
                    continue
                src_event = recent_slice[idx - 1]
                src_global = src_event.get("id")
                if not isinstance(src_global, int):
                    continue
                external_edges.append(
                    {
                        "src": src_global,
                        "dst": next_id,
                        "kind": str(ext.get("kind", "")),
                        "reason": str(ext.get("reason", "")),
                        "src_turns": list(src_event.get("source_turns") or []),
                        "dst_turns": list(ev.get("source_turns") or []),
                        "cited_entities": list(ext.get("cited_entities") or []),
                        "cited_quote": str(ext.get("cited_quote", "") or ""),
                    }
                )

        relabelled_edges: list[dict[str, Any]] = []
        for ed in new_edges:
            src = ed.get("src")
            dst = ed.get("dst")
            new_src = local_to_global.get(src) if isinstance(src, int) else None
            new_dst = local_to_global.get(dst) if isinstance(dst, int) else None
            if new_src is None or new_dst is None:
                relabelled_edges.append(ed)
                continue
            relabelled_edges.append({**ed, "src": new_src, "dst": new_dst})

        cum_events = [*cum_events, *relabelled_events]
        cum_edges = [*cum_edges, *relabelled_edges, *external_edges]
        snapshots.append(
            GraphSnapshot(
                after_extractor_firing=fr.sequence,
                turn_index=fr.turn_index,
                events=cum_events,
                edges=cum_edges,
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


def _main_agent_messages(
    auditor_records: list[ReplayRecord],
    extractor_records: list[ReplayRecord],
) -> list[dict[str, Any]]:
    """Reconstruct the most complete main-agent trajectory.

    Two-stage stitch:

    1. **Base**: the latest ``compose_kwargs.trajectory_snapshot`` from
       a successful auditor firing. The auditor's snapshot is the
       authoritative serialised trajectory at that firing — every
       message up to ``turn_index`` is present.
    2. **Tail**: any extractor firing whose ``turn_index`` is greater
       than the base's last covered turn contributes its
       ``payload.new_turns`` window. Extractor records are walked in
       chronological order; messages whose ``index`` is already in the
       base are skipped (idempotent merge).

    Without the tail stitch, the case directory captures only up to
    the last auditor firing — trailing main-agent turns disappear
    when the run ends mid-auditor-interval, which is the common case.
    """
    base: list[dict[str, Any]] = []
    for rec in reversed(auditor_records):
        if rec.status == "ok":
            snap = (rec.compose_kwargs or {}).get("trajectory_snapshot")
            if isinstance(snap, list):
                base = list(snap)
                break
    if not base:
        # Fall back to any snapshot if no ok auditor firing.
        for rec in reversed(auditor_records):
            snap = (rec.compose_kwargs or {}).get("trajectory_snapshot")
            if isinstance(snap, list):
                base = list(snap)
                break

    # Highest message index already covered by the base.
    seen_indices: set[int] = set()
    max_index = -1
    for msg in base:
        idx = msg.get("index") if isinstance(msg, dict) else None
        if isinstance(idx, int):
            seen_indices.add(idx)
            if idx > max_index:
                max_index = idx

    # Append from extractor records strictly after the base, in time
    # order. ``payload.new_turns`` is the window the extractor saw
    # since the prior cursor — concatenating in turn-order yields a
    # continuous tail.
    tail: list[dict[str, Any]] = []
    for rec in sorted(extractor_records, key=lambda r: (r.ts_ns, r.turn_index)):
        new_turns = (rec.payload or {}).get("new_turns")
        if not isinstance(new_turns, list):
            continue
        for msg in new_turns:
            if not isinstance(msg, dict):
                continue
            idx = msg.get("index")
            if isinstance(idx, int):
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)
            tail.append(msg)

    return base + tail


def collect_case(
    *,
    replay_path: Path,
    meta_path: Path | None = None,
    sample_id_override: str | None = None,
    dataset_name_override: str | None = None,
    dataset_path_override: str | None = None,
) -> CaseData:
    """Build a :class:`CaseData` from one replay sidecar.

    Sample-id resolution precedence: explicit ``sample_id_override`` >
    meta sidecar > ``None`` (case_id then derives from
    ``root_session_id``). The overrides let the CLI inject case
    metadata when the run did not mount ``llmharness.distill.binding``
    (e.g. rca llm-eval runs).
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
    sample_id = (
        sample_id_override
        if sample_id_override
        else (meta_obj.sample_id if meta_obj is not None else None)
    )
    dataset_name = (
        dataset_name_override
        if dataset_name_override
        else (meta_obj.dataset_name if meta_obj is not None else None)
    )
    dataset_path = (
        dataset_path_override
        if dataset_path_override
        else (meta_obj.dataset_path if meta_obj is not None else None)
    )

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
        main_agent_messages=_main_agent_messages(auditor_records, extractor_records),
        extractor_firings=extractor_firings,
        auditor_firings=auditor_firings,
        graph_snapshots=_accumulate_graph(extractor_firings),
        verdicts=verdicts,
    )


__all__ = ["collect_case"]
