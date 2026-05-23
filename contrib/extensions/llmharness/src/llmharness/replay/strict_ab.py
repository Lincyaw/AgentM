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

from ..audit._runner import AuditorSettings, ExtractorSettings
from .offline_driver import replay_pipeline_over_trajectory
from .record import ReplayRecord, iter_records, write_record

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


def _infer_interval(turn_indices: list[int], default: int = 5) -> int:
    """Derive the runner cadence from a list of recorded turn indices.

    The control sidecar's ``phase=='extractor'`` records were emitted
    every ``k`` turns by the live runner; the deltas between turn
    indices are therefore the cadence. Inspecting up to three records
    is sufficient — if the first three deltas don't agree the records
    are stitched from a non-uniform run, and we fall back to ``default``
    rather than picking one arbitrarily.
    """
    if len(turn_indices) < 2:
        return default
    deltas: list[int] = []
    for i in range(1, min(len(turn_indices), 4)):
        d = turn_indices[i] - turn_indices[i - 1]
        if d > 0:
            deltas.append(d)
    if not deltas:
        return default
    first = deltas[0]
    return first if all(d == first for d in deltas) else default


def _settings_from_sidecar(
    control_replay_path: Path,
) -> tuple[ExtractorSettings, AuditorSettings, int, int]:
    """Extract per-firing compose state + cadence from the control sidecar.

    Returns ``(extractor_settings, auditor_settings, extractor_interval,
    audit_interval)``. The first record of each phase carries the
    compose_kwargs the live adapter would have used; subsequent
    records reuse the same compose state modulo cadence so reading
    the head record is sufficient. Missing records → module defaults
    (empty base prompt, default cadence 5).
    """
    extractor_records: list[ReplayRecord] = []
    auditor_records: list[ReplayRecord] = []
    for rec in iter_records(control_replay_path):
        if rec.phase == "extractor":
            extractor_records.append(rec)
        elif rec.phase == "auditor":
            auditor_records.append(rec)

    ext_settings = (
        ExtractorSettings.from_compose_kwargs(extractor_records[0].compose_kwargs)
        if extractor_records
        else ExtractorSettings.from_compose_kwargs({})
    )
    aud_settings = (
        AuditorSettings.from_compose_kwargs(auditor_records[0].compose_kwargs)
        if auditor_records
        else AuditorSettings.empty()
    )
    ext_interval = _infer_interval([r.turn_index for r in extractor_records])
    aud_interval = _infer_interval([r.turn_index for r in auditor_records])
    return ext_settings, aud_settings, ext_interval, aud_interval


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

    Thin caller of :func:`replay_pipeline_over_trajectory` (P3 of
    ``.claude/designs/harness-runner.md``). Pulls extractor / auditor
    settings + cadence from the control sidecar's first record of each
    phase, drives the runner across the control trajectory, and
    collects synthetic auditor :class:`ReplayRecord` s from each
    fired :class:`StepResult`. Missing control sidecar → empty
    :class:`OfflineAuditRun`.
    """
    if not control_replay_path.exists():
        return OfflineAuditRun(reminder=None, records=[])

    ext_settings, aud_settings, ext_interval, aud_interval = _settings_from_sidecar(
        control_replay_path
    )

    run = await replay_pipeline_over_trajectory(
        messages=list(control_messages),
        cwd=cwd,
        root_session_id=control_session_log_id,
        provider=provider,
        extractor_settings=ext_settings,
        auditor_settings=aud_settings,
        extractor_interval=ext_interval,
        audit_interval=aud_interval,
        enable_auditor=True,
        stop_on_first_surface=stop_on_first_surface,
        sidecar_path=None,
    )

    auditor_records: list[ReplayRecord] = [
        step.auditor_record
        for step in run.all_step_results
        if step.auditor_record is not None
    ]

    reminder: ReminderCandidate | None = None
    for record in auditor_records:
        if not isinstance(record.output, dict):
            continue
        if not record.output.get("surface_reminder"):
            continue
        text = record.output.get("reminder_text")
        if not isinstance(text, str) or not text.strip():
            continue
        reminder = ReminderCandidate(
            turn_index=int(record.turn_index),
            text=text,
            record=record,
        )
        break

    return OfflineAuditRun(reminder=reminder, records=auditor_records)


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

    If the branch produces no post-reminder extractor records (e.g.
    it terminated immediately on the seeded reminder), the sidecar
    carries only the control prefix + surfaced reminder — no fallback
    write of branch records with ``turn_index <= reminder.turn_index``,
    because those collide with the control prefix and corrupt any
    downstream turn-keyed view. Returns the output path.
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

    # Strictly post-cut: ``turn_index > reminder.turn_index`` only.
    # An empty branch tail (early termination, no post-reminder
    # extractor firing) is a valid outcome — the sidecar then carries
    # only the control prefix + surfaced reminder, and consumers can
    # distinguish "no branch-side audit signal" from "branch disagreed
    # with control" cleanly. We do NOT fall back to dumping branch
    # records with turn_index <= reminder.turn_index: those collide
    # with the control prefix's turn-keyed records and corrupt any
    # downstream ``{turn_index: record}`` view of the sidecar.
    for record in branch_records:
        if record.phase != "extractor":
            continue
        if int(record.turn_index) <= reminder.turn_index:
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
