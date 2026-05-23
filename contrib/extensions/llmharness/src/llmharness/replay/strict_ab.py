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
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from itertools import pairwise
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage

from ..audit._runner import AuditorSettings, ExtractorSettings
from . import offline_driver as _offline_driver
from .offline_driver import OfflineRunResult
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


_EXTRACTOR_FIRING_STATUSES = frozenset(
    {"ok", "no_call", "spawn_error", "prompt_error"}
)


def _infer_extractor_interval(
    records: list[ReplayRecord], *, default: int = 5
) -> int:
    """Derive the extractor cadence from recorded extractor firings.

    The live runner enforces ``extractor_due = (turn_count %
    extractor_k) == 0 or auditor_due`` AND ``auditor_due = (turn_count
    % k) == 0`` on every ``_on_turn_end``, so the extractor fires on
    every cadence boundary unconditionally — auditor firings are a
    strict subset of extractor firings. Auditor records, on the other
    hand, can have non-uniform turn-index deltas because the runner
    skips an auditor firing when the preceding extractor held the
    cursor (see ``_runner.HarnessRunner._last_extractor_held_cursor``).
    Inferring cadence from auditor records is therefore unsafe; the
    fix is to derive it from extractor records only.

    Filters ``records`` to ``phase=='extractor'`` and statuses where a
    firing attempt actually occurred (``ok``, ``no_call``,
    ``spawn_error``, ``prompt_error``). Returns ``gcd(*deltas)`` of
    consecutive turn-index deltas — for uniform cadence this is the
    cadence itself; for sidecars stitched across non-uniform runs
    ``gcd`` is the largest interval that divides every observed
    delta. Falls back to ``default`` only when fewer than two firings
    are available.
    """
    extractor_firings = [
        r
        for r in records
        if r.phase == "extractor" and r.status in _EXTRACTOR_FIRING_STATUSES
    ]
    if len(extractor_firings) < 2:
        return default
    extractor_firings.sort(key=lambda r: r.turn_index)
    deltas: list[int] = []
    for prev, cur in pairwise(extractor_firings):
        d = cur.turn_index - prev.turn_index
        if d > 0:
            deltas.append(d)
    if not deltas:
        return default
    inferred = reduce(math.gcd, deltas)
    return inferred if inferred > 0 else default


def _settings_from_sidecar(
    control_replay_path: Path,
) -> tuple[ExtractorSettings, AuditorSettings, int]:
    """Extract per-firing compose state + cadence from the control sidecar.

    Returns ``(extractor_settings, auditor_settings, extractor_interval)``.
    The first record of each phase carries the compose_kwargs the live
    adapter would have used; subsequent records reuse the same compose
    state modulo cadence so reading the head record is sufficient.
    Missing records → module defaults (empty base prompt, default
    cadence 5).

    In ``rca:harness.sync*`` variants the extractor and auditor share
    the same cadence (auditor firings are a strict subset of extractor
    firings on the same boundary — see
    :func:`_infer_extractor_interval`). The single inferred cadence is
    therefore the right value to pass for both ``extractor_interval``
    and ``audit_interval`` in :func:`replay_pipeline_over_trajectory`.
    """
    records = list(iter_records(control_replay_path))
    extractor_records = [r for r in records if r.phase == "extractor"]
    auditor_records = [r for r in records if r.phase == "auditor"]

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
    interval = _infer_extractor_interval(records)
    _logger.info(
        "strict_ab: inferred cadence interval=%d from %d extractor record(s) "
        "in %s",
        interval,
        len(extractor_records),
        control_replay_path,
    )
    return ext_settings, aud_settings, interval


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

    ext_settings, aud_settings, interval = _settings_from_sidecar(
        control_replay_path
    )

    run = await _offline_driver.replay_pipeline_over_trajectory(
        messages=list(control_messages),
        cwd=cwd,
        root_session_id=control_session_log_id,
        provider=provider,
        extractor_settings=ext_settings,
        auditor_settings=aud_settings,
        extractor_interval=interval,
        audit_interval=interval,
        enable_auditor=True,
        stop_on_first_surface=stop_on_first_surface,
        sidecar_path=None,
    )
    return _offline_run_to_audit(run)


def _offline_run_to_audit(offline_run: OfflineRunResult) -> OfflineAuditRun:
    """Convert a fresh OfflineRunResult into the OfflineAuditRun shape.

    Recovers a ReminderCandidate from the synthesised auditor records by
    inspecting each record's output for ``surface_reminder`` /
    ``reminder_text`` — the same mining logic
    :func:`run_offline_auditor_over_control` uses on a sidecar-driven
    walk.
    """
    auditor_records: list[ReplayRecord] = [
        step.auditor_record
        for step in offline_run.all_step_results
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


async def run_offline_auditor_over_trajectory(
    *,
    messages: Sequence[AgentMessage],
    cwd: str,
    root_session_id: str,
    provider: tuple[str, dict[str, Any]] | None,
    extractor_settings: ExtractorSettings,
    auditor_settings: AuditorSettings,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    stop_on_first_surface: bool = True,
    sidecar_path: Path | None = None,
) -> OfflineAuditRun:
    """Drive the offline pipeline over a raw trajectory and surface the
    first reminder candidate, returning the same OfflineAuditRun shape
    that run_offline_auditor_over_control yields for sidecar-driven
    input.

    Use when the control rollout has no in-line extractor (e.g. pure
    ``rca:baseline`` runs). The runner synthesises extractor + auditor
    records on the fly, writes them to ``sidecar_path`` if provided,
    and the returned ``records`` are pulled from each step's
    ``auditor_record``.
    """
    offline_run = await _offline_driver.replay_pipeline_over_trajectory(
        messages=list(messages),
        cwd=cwd,
        root_session_id=root_session_id,
        provider=provider,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=extractor_interval,
        audit_interval=audit_interval,
        enable_auditor=True,
        stop_on_first_surface=stop_on_first_surface,
        sidecar_path=sidecar_path,
    )
    return _offline_run_to_audit(offline_run)


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
    "run_offline_auditor_over_trajectory",
    "strict_ab_replay_path",
    "write_strict_ab_replay",
]
