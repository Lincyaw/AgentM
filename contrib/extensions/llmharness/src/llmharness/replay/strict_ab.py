"""Strict-A/B replay orchestration.

A strict-A/B fork experiment compares a *control* main-agent trajectory
(no auditor reminders) with a *branch* trajectory that re-runs from the
same prefix after a single seeded reminder. Strict means: the control
session must not mount the auditor — judgements are reconstructed
**offline** from the extractor-only replay sidecar so the control
prefix is immutable across runs.

Two entry points:

* :func:`run_offline_auditor_over_control` — walks the control's
  extractor records, replays the auditor against each cumulative graph
  snapshot, and returns the first record that surfaces a reminder
  (plus, optionally, every silent auditor record collected along the
  way).
* :func:`write_strict_ab_replay` — materializes a unified replay
  sidecar for the branch by stitching the control's
  extractor/auditor prefix (up to and including the surfaced-reminder
  turn) with the branch's extractor/auditor tail. The case viewer
  reads this single file and sees a clean A/B comparison.

These helpers do not own scenario-name policy — callers tell us which
control scenario to run by passing the already-collected control
trajectory. The naming convention for the output sidecar mirrors
:func:`llmharness.replay.record.replay_log_path`: it is named after the
branch session's log id with a ``.strict_ab.jsonl`` suffix.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage

from ..adapters.agentm import flatten_assistant_blocks, serialize_full_trajectory
from ..audit.auditor.output import AuditorOutputError, RawVerdictOutput
from ..audit.phase import merge_to_phases
from ..schema import Edge, Event
from .record import ReplayRecord, iter_records, now_ns, write_record
from .runner import replay_auditor_record

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReminderCandidate:
    """An auditor verdict that surfaced a reminder during offline replay.

    ``turn_index`` is the main-agent turn at which the auditor would
    have fired (matches the extractor record it was paired with).
    ``record`` is the auditor :class:`ReplayRecord` that carried the
    surfaced ``reminder_text`` — useful for downstream sidecar writes.
    """

    turn_index: int
    text: str
    record: ReplayRecord


@dataclass(frozen=True)
class OfflineAuditRun:
    """Outcome of :func:`run_offline_auditor_over_control`.

    ``reminder`` is the first surfaced reminder under the configured
    stop policy (None when no auditor firing surfaced one).
    ``records`` is every auditor record produced during the walk —
    including silent verdicts — so callers can persist a complete
    audit-replay alongside the control trajectory.
    """

    reminder: ReminderCandidate | None
    records: list[ReplayRecord] = field(default_factory=list)


def strict_ab_replay_path(cwd: str | Path, branch_session_log_id: str) -> Path:
    """Canonical sidecar path for the unified strict-A/B replay.

    Mirrors :func:`llmharness.replay.record.replay_log_path` but with a
    ``.strict_ab.jsonl`` suffix keyed off the branch session log id.
    Branch keyed (not control keyed) so the case viewer can find the
    file via the branch's ``audit_replay_path`` metadata field.
    """

    return Path(cwd) / ".agentm" / "audit_replay" / f"{branch_session_log_id}.strict_ab.jsonl"


def _coerce_schema_list(cls: Any, items: Any) -> list[Any]:
    """Best-effort dict-to-dataclass coercion for events / edges.

    Replay records carry serialized dicts; the auditor composer wants
    ``Event`` / ``Edge`` dataclasses. Skip anything that fails to
    deserialize rather than crashing the whole walk over one bad row.
    """

    out: list[Any] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(cls.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def run_offline_auditor_over_control(
    *,
    control_replay_path: Path,
    control_session_log_id: str,
    control_messages: Sequence[AgentMessage],
    cwd: str,
    provider: tuple[str, dict[str, Any]] | None,
    stop_on_first_surface: bool = True,
) -> OfflineAuditRun:
    """Replay the auditor side-channel against a fixed control trajectory.

    Walks every ``ok`` extractor record on the control's replay
    sidecar in (ts_ns, turn_index) order. For each one we (1) fold the
    extracted events / edges into a cumulative graph, (2) compose the
    same auditor payload the live adapter would build at that turn,
    (3) re-run the auditor via :func:`replay_auditor_record`, and
    (4) collect the resulting auditor record. If a verdict's
    ``surface_reminder`` is set and ``reminder_text`` is non-empty,
    that turn becomes the candidate.

    When ``stop_on_first_surface`` is true (default) we return as soon
    as a candidate appears. When false, the walk continues to the end
    and ``OfflineAuditRun.records`` contains every silent verdict too
    — useful for branch-side audit replays where the case viewer
    wants every firing, not just the first.

    Missing control sidecar → empty :class:`OfflineAuditRun`. Schema
    coercion failures on individual records are skipped (see
    :func:`_coerce_schema_list`).
    """

    if not control_replay_path.exists():
        return OfflineAuditRun(reminder=None, records=[])

    events: list[Event] = []
    edges: list[Edge] = []
    phases: list[Any] = []
    recent_verdicts: list[dict[str, Any]] = []
    continuation_notes: list[str] = []
    auditor_records: list[ReplayRecord] = []

    extractor_records = [
        rec
        for rec in iter_records(control_replay_path)
        if rec.phase == "extractor" and rec.status == "ok" and rec.output
    ]
    extractor_records.sort(key=lambda rec: (rec.ts_ns, rec.turn_index))

    for extractor_record in extractor_records:
        output = extractor_record.output or {}
        new_events = _coerce_schema_list(Event, output.get("events") or [])
        new_edges = _coerce_schema_list(Edge, output.get("edges") or [])
        events.extend(new_events)
        edges.extend(new_edges)
        phases.extend(merge_to_phases(new_events))

        cut = min(
            max(extractor_record.turn_index + 1, 0), len(control_messages)
        )
        trajectory_snapshot = serialize_full_trajectory(
            list(control_messages[:cut])
        )
        compose_kwargs: dict[str, Any] = {
            "base_prompt": None,
            "cards_tools_config": {},
            "observability_config": {},
            "trajectory_snapshot": trajectory_snapshot,
            "events": [ev.to_dict() for ev in events],
            "edges": [ed.to_dict() for ed in edges],
            "phases": [ph.to_dict() for ph in phases],
            "findings": [],
            "check_errors": {},
            "continuation_notes": list(continuation_notes),
            "summary_threshold": 30,
            "tools": ["submit_verdict"],
        }
        payload: dict[str, Any] = {
            "graph": [ev.to_dict() for ev in events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(continuation_notes),
        }
        replay_input = ReplayRecord(
            phase="auditor",
            turn_index=extractor_record.turn_index,
            root_session_id=control_session_log_id,
            ts_ns=(extractor_record.ts_ns or now_ns()) + 1,
            compose_kwargs=compose_kwargs,
            payload=payload,
            provider=None,
            output=None,
            status="ok",
        )
        phase_result = await replay_auditor_record(
            replay_input,
            cwd=cwd,
            provider_override=provider,
        )
        verdict_dict: dict[str, Any] | None = None
        if phase_result.status == "ok" and isinstance(phase_result.output, dict):
            try:
                verdict_dict = (
                    RawVerdictOutput.from_dict(phase_result.output)
                    .to_verdict()
                    .to_dict()
                )
            except AuditorOutputError:
                verdict_dict = None
        auditor_record = ReplayRecord(
            phase="auditor",
            turn_index=extractor_record.turn_index,
            root_session_id=control_session_log_id,
            ts_ns=replay_input.ts_ns,
            compose_kwargs=compose_kwargs,
            payload=payload,
            provider=[provider[0], provider[1]] if provider else None,
            output=verdict_dict,
            status=phase_result.status if verdict_dict is not None else "no_call",
            error=phase_result.error,
            latency_ms=phase_result.latency_ms,
            raw_assistant_messages=flatten_assistant_blocks(phase_result.messages),
        )
        auditor_records.append(auditor_record)

        if verdict_dict is not None:
            recent_verdicts.append(verdict_dict)
            raw_notes = verdict_dict.get("continuation_notes")
            continuation_notes = (
                [str(n) for n in raw_notes if isinstance(n, str)]
                if isinstance(raw_notes, list)
                else []
            )

        if not verdict_dict or not verdict_dict.get("surface_reminder"):
            continue
        text = verdict_dict.get("reminder_text")
        if not isinstance(text, str) or not text.strip():
            continue
        reminder = ReminderCandidate(
            turn_index=extractor_record.turn_index,
            text=text,
            record=auditor_record,
        )
        if stop_on_first_surface:
            return OfflineAuditRun(reminder=reminder, records=auditor_records)

    first_reminder = next(
        (
            ReminderCandidate(
                turn_index=int(record.turn_index),
                text=str((record.output or {}).get("reminder_text", "")),
                record=record,
            )
            for record in auditor_records
            if isinstance(record.output, dict)
            and record.output.get("surface_reminder")
            and str(record.output.get("reminder_text", "")).strip()
        ),
        None,
    )
    return OfflineAuditRun(reminder=first_reminder, records=auditor_records)


def _rebind_record(record: ReplayRecord, *, root_session_id: str) -> ReplayRecord:
    """Return a copy of ``record`` re-keyed under a new root_session_id.

    The strict-A/B sidecar is keyed off the *branch* session id but
    composed from control records, so every entry needs its
    ``root_session_id`` rewritten while everything else (timing,
    payload, output) stays verbatim.
    """

    return replace(record, root_session_id=root_session_id)


def write_strict_ab_replay(
    *,
    control_replay_path: Path,
    branch_replay_path: Path,
    branch_session_log_id: str,
    offline_auditor_records: Sequence[ReplayRecord],
    branch_auditor_records: Sequence[ReplayRecord] | None,
    reminder: ReminderCandidate,
    out_path: Path,
) -> Path:
    """Materialize the unified strict-A/B replay sidecar.

    The composition is::

        [control extractor + offline auditor]  for turns <= reminder.turn_index
        [branch  extractor + branch  auditor]  for turns >  reminder.turn_index

    All entries are rewritten to live under ``branch_session_log_id``
    so the case viewer can find them via the branch's
    ``audit_replay_path``. Silent auditor verdicts are persisted too so
    the viewer sees one auditor firing per extractor firing — only the
    surfaced-reminder turn is special.

    If the branch sidecar happens to be empty of post-reminder
    extractor records (e.g. the branch terminated immediately on the
    seeded reminder), the entire branch sidecar is written verbatim
    as a fallback so the viewer still has *something* on the branch
    side. Returns the output path.
    """

    if out_path.exists():
        out_path.unlink()

    control_records = list(iter_records(control_replay_path)) if control_replay_path.exists() else []
    branch_records = list(iter_records(branch_replay_path)) if branch_replay_path.exists() else []
    control_auditors_by_turn = {
        int(record.turn_index): record for record in offline_auditor_records
    }
    branch_auditors_by_turn = {
        int(record.turn_index): record
        for record in (branch_auditor_records or [])
    }

    for record in control_records:
        if record.phase != "extractor":
            continue
        if int(record.turn_index) > reminder.turn_index:
            continue
        write_record(
            out_path,
            _rebind_record(record, root_session_id=branch_session_log_id),
        )
        auditor = control_auditors_by_turn.get(int(record.turn_index))
        if auditor is not None:
            write_record(
                out_path,
                _rebind_record(auditor, root_session_id=branch_session_log_id),
            )

    branch_tail_written = False
    for record in branch_records:
        if record.phase != "extractor":
            continue
        if int(record.turn_index) <= reminder.turn_index:
            continue
        branch_tail_written = True
        write_record(
            out_path,
            _rebind_record(record, root_session_id=branch_session_log_id),
        )
        auditor = branch_auditors_by_turn.get(int(record.turn_index))
        if auditor is not None:
            write_record(
                out_path,
                _rebind_record(auditor, root_session_id=branch_session_log_id),
            )

    if not branch_tail_written:
        # Branch produced no post-reminder extractor records — emit
        # whatever extractor / auditor pairs it does have so the
        # viewer can still show the branch side.
        for record in branch_records:
            if record.phase != "extractor":
                continue
            write_record(
                out_path,
                _rebind_record(record, root_session_id=branch_session_log_id),
            )
            auditor = branch_auditors_by_turn.get(int(record.turn_index))
            if auditor is not None:
                write_record(
                    out_path,
                    _rebind_record(auditor, root_session_id=branch_session_log_id),
                )

    return out_path


__all__ = [
    "OfflineAuditRun",
    "ReminderCandidate",
    "run_offline_auditor_over_control",
    "strict_ab_replay_path",
    "write_strict_ab_replay",
]
