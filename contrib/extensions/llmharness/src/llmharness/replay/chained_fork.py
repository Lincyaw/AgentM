"""Chained-fork experiment driver.

A *chained-fork* run repeatedly seeds the agent with a surfaced
reminder. Segment 0 (the "control") runs to completion (or until the
auditor first surfaces a reminder under ``stop_on_first_surface``).
Each subsequent segment k forks from the **previous** segment at the
turn where its auditor surfaced a reminder, re-seeds that reminder
into the child session, and runs until either the next reminder
surfaces or the trajectory ends. The chain terminates when either no
new reminder surfaces or ``max_interventions`` segments have been
appended.

Compared with the deleted strict-A/B driver, this driver:

* Threads :class:`CumulativeAuditState` across segment boundaries
  (segment k+1 starts with segment k's auditor state) — see
  :func:`replay_pipeline_over_trajectory`'s ``seed_cumulative`` /
  ``start_turn`` knobs.
* Emits one sidecar (``<final_branch_sid>.chained.jsonl``) covering
  the full chain instead of separate control + branch + strict-A/B
  files.
* Has no notion of a "control scenario" distinct from the branch
  scenario — every segment is the same scenario, only the seeded
  prefix + reminder differ.

See ``.claude/designs/harness-runner.md`` §3 (P4+) for the rationale.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from agentm.core.abi.messages import AgentMessage

from ..audit._offline_seams import InMemorySink, StandaloneChildRunner
from ..audit._runner import (
    AuditorSettings,
    CumulativeAuditState,
    ExtractorSettings,
    StepResult,
)
from ..schema import Reminder
from .offline_driver import replay_pipeline_over_trajectory
from .record import ReplayRecord, write_record

_logger = logging.getLogger(__name__)

__all__ = [
    "ChainSegment",
    "ChainSegmentPayload",
    "ChainedForkExperiment",
    "SessionFactory",
    "chained_replay_path",
    "run_chained_fork_experiment",
    "write_chained_replay",
]


class ChainSegmentPayload(Protocol):
    """The seam between the experiment driver and the host session.

    A :class:`SessionFactory` returns one of these per segment. Anything
    that exposes ``session_log_id: str`` and
    ``final_messages: list[AgentMessage]`` qualifies — notably the rca
    eval driver's ``_SessionRun`` dataclass matches structurally so the
    factory can return it directly.
    """

    session_log_id: str
    final_messages: list[AgentMessage]


SessionFactory = Callable[..., Awaitable[ChainSegmentPayload]]
"""Coroutine: ``(*, initial_messages, seed_reminder_text) -> ChainSegmentPayload``.

The control segment is invoked with ``initial_messages=None`` and
``seed_reminder_text=None``; each branch segment k receives the parent
trajectory prefix and the reminder text harvested from segment k-1.
"""


@dataclass(frozen=True)
class ChainSegment:
    """One segment of a chained-fork experiment.

    ``segment_index == 0`` is the control. ``fork_turn_index`` is the
    turn in the *parent* segment's trajectory where this segment forked
    from (None for the control). ``surfaced_reminder_turn`` is the turn
    in *this* segment's own trajectory at which the auditor next
    surfaced a reminder, or ``None`` if the auditor stayed silent for
    the rest of the trajectory (chain terminator).
    """

    segment_index: int
    payload: ChainSegmentPayload
    seeded_reminder: Reminder | None
    fork_turn_index: int | None
    surfaced_reminder_turn: int | None
    step_results: list[StepResult] = field(default_factory=list)


@dataclass(frozen=True)
class ChainedForkExperiment:
    """Outcome of one :func:`run_chained_fork_experiment` invocation."""

    segments: list[ChainSegment]
    chained_replay_path: Path | None

    @property
    def control(self) -> ChainSegment:
        return self.segments[0]

    @property
    def final(self) -> ChainSegment:
        return self.segments[-1]


def chained_replay_path(cwd: str | Path, branch_session_log_id: str) -> Path:
    """Canonical sidecar path for a chained-fork replay.

    Mirrors :func:`llmharness.replay.record.replay_log_path` but with a
    ``.chained.jsonl`` suffix keyed off the *final* branch session log
    id so the case viewer can find the file via that branch's
    ``audit_replay_path`` metadata.
    """
    return (
        Path(cwd)
        / ".agentm"
        / "audit_replay"
        / f"{branch_session_log_id}.chained.jsonl"
    )


def _rebind_record(record: ReplayRecord, *, root_session_id: str) -> ReplayRecord:
    """Return a copy of ``record`` re-keyed under a new root_session_id.

    The chained-fork sidecar is keyed off the *final* branch session id
    but composed from records produced under several different session
    ids (one per segment). Every entry's ``root_session_id`` is
    rewritten while everything else (timing, payload, output) stays
    verbatim.
    """
    return replace(record, root_session_id=root_session_id)


async def run_chained_fork_experiment(
    *,
    session_factory: SessionFactory,
    cwd: str,
    provider: tuple[str, dict[str, Any]] | None,
    extractor_settings: ExtractorSettings,
    auditor_settings: AuditorSettings,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    max_interventions: int = 10,
    out_path: Path | None = None,
    sink: InMemorySink | None = None,
    child: StandaloneChildRunner | None = None,
) -> ChainedForkExperiment:
    """Drive a chained-fork experiment to completion.

    Algorithm (in order):

    1. Run the control segment via ``session_factory(initial_messages=None,
       seed_reminder_text=None)``.
    2. Replay the audit pipeline offline over the control trajectory with
       ``stop_on_first_surface=True`` and capture the cumulative state.
    3. If no reminder surfaced, return a single-segment experiment.
    4. Otherwise, for up to ``max_interventions`` iterations: fork the
       parent trajectory at ``next_reminder.turn_index + 1`` (slice
       ``[:turn_index + 1]`` matches the live ``_fork_prefix_messages``
       inclusive cut), call the factory with the prefix + reminder
       text, replay the pipeline against the resulting trajectory
       starting at ``start_turn=turn_index + 1`` with the prior
       segment's cumulative state seeded in. Stop when a segment
       produces no reminder.
    5. Materialise the chained replay sidecar (if ``chained_replay_path``
       is None, derive it from the final segment's session_log_id).

    The ``sink`` / ``child`` parameters are exposed for tests that
    inject stubs while still exercising the real driver.
    """
    control_payload = await session_factory(
        initial_messages=None, seed_reminder_text=None
    )
    control_run = await replay_pipeline_over_trajectory(
        messages=control_payload.final_messages,
        cwd=cwd,
        root_session_id=control_payload.session_log_id,
        provider=provider,
        extractor_settings=extractor_settings,
        auditor_settings=auditor_settings,
        extractor_interval=extractor_interval,
        audit_interval=audit_interval,
        enable_auditor=True,
        stop_on_first_surface=True,
        sidecar_path=None,
        sink=sink,
        child=child,
        seed_cumulative=None,
        start_turn=1,
    )
    control_surface_turn = (
        _last_audited_turn(control_run.all_step_results)
        if control_run.reminder is not None
        else None
    )
    segments: list[ChainSegment] = [
        ChainSegment(
            segment_index=0,
            payload=control_payload,
            seeded_reminder=None,
            fork_turn_index=None,
            surfaced_reminder_turn=control_surface_turn,
            step_results=list(control_run.all_step_results),
        )
    ]

    if control_run.reminder is None:
        return ChainedForkExperiment(segments=segments, chained_replay_path=None)

    cumulative: CumulativeAuditState = control_run.state
    parent_messages = control_payload.final_messages
    next_reminder: Reminder | None = control_run.reminder
    # The ``Reminder`` schema doesn't carry a turn_index; the runner
    # surfaces ``Reminder(text=...)`` and the surfacing turn equals the
    # last audited turn captured in the step results.
    next_fork_turn = control_surface_turn

    for i in range(max_interventions):
        if next_reminder is None or next_fork_turn is None:
            break
        # Fork prefix: turns [0..next_fork_turn] inclusive. The new
        # segment begins replaying from turn next_fork_turn + 1.
        prefix = list(parent_messages[: next_fork_turn + 1])
        branch_payload = await session_factory(
            initial_messages=prefix,
            seed_reminder_text=next_reminder.text,
        )
        branch_run = await replay_pipeline_over_trajectory(
            messages=branch_payload.final_messages,
            cwd=cwd,
            root_session_id=branch_payload.session_log_id,
            provider=provider,
            extractor_settings=extractor_settings,
            auditor_settings=auditor_settings,
            extractor_interval=extractor_interval,
            audit_interval=audit_interval,
            enable_auditor=True,
            stop_on_first_surface=True,
            sidecar_path=None,
            sink=sink,
            child=child,
            seed_cumulative=cumulative,
            start_turn=next_fork_turn + 1,
        )
        branch_surface_turn = (
            _last_audited_turn(branch_run.all_step_results)
            if branch_run.reminder is not None
            else None
        )
        segments.append(
            ChainSegment(
                segment_index=i + 1,
                payload=branch_payload,
                seeded_reminder=next_reminder,
                fork_turn_index=next_fork_turn,
                surfaced_reminder_turn=branch_surface_turn,
                step_results=list(branch_run.all_step_results),
            )
        )
        if branch_run.reminder is None:
            break
        cumulative = branch_run.state
        parent_messages = branch_payload.final_messages
        next_reminder = branch_run.reminder
        next_fork_turn = branch_surface_turn

    resolved_out: Path = (
        out_path
        if out_path is not None
        else chained_replay_path(cwd, segments[-1].payload.session_log_id)
    )
    sidecar = write_chained_replay(segments, out_path=resolved_out)
    return ChainedForkExperiment(segments=segments, chained_replay_path=sidecar)


def _last_audited_turn(step_results: list[StepResult]) -> int | None:
    """Return the turn index of the last auditor firing in ``step_results``.

    The auditor record's ``turn_index`` is ``len(messages) - 1`` at the
    cadence boundary on which the firing occurred. We recover it from
    the step's ``auditor_record`` rather than from the step's position
    in the list because the loop emits one step per turn whether the
    auditor fired or not.
    """
    for step in reversed(step_results):
        if step.auditor_record is not None:
            return int(step.auditor_record.turn_index)
    return None


def write_chained_replay(
    segments: list[ChainSegment],
    *,
    out_path: Path,
) -> Path:
    """Materialise an N-segment chained replay sidecar.

    For each segment k, emit the extractor + auditor records captured
    in ``step_results`` whose ``turn_index`` falls inside the segment's
    half-open interval ``[turn_lo_k, turn_hi_k]``:

    * segment 0 (control): ``turn_lo = 0``,
      ``turn_hi = surfaced_reminder_turn`` if the chain extends past
      this segment, else the last turn the auditor saw.
    * segment k > 0: ``turn_lo = fork_turn_index + 1``,
      ``turn_hi = surfaced_reminder_turn`` if the chain extends past
      this segment, else the last turn the auditor saw.

    All records are rebound to ``segments[-1].payload.session_log_id``
    so the case viewer joins them against the final branch's
    trajectory regardless of which segment they came from.

    Returns the output path. If ``out_path`` already exists it is
    truncated first so re-running the experiment overwrites cleanly.
    """
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_sid = segments[-1].payload.session_log_id

    for k, segment in enumerate(segments):
        if k == 0:
            turn_lo = 0
        else:
            assert segment.fork_turn_index is not None, (
                "non-control segments must have fork_turn_index set"
            )
            turn_lo = segment.fork_turn_index + 1
        # Upper bound: if a successor segment exists, this segment's
        # window ends at its surfaced_reminder_turn (inclusive — the
        # firing that triggered the next fork still belongs to this
        # segment). Otherwise, accept every record this segment
        # produced (chain terminator).
        is_terminator = k == len(segments) - 1
        if not is_terminator and segment.surfaced_reminder_turn is not None:
            turn_hi: int | None = segment.surfaced_reminder_turn
        else:
            turn_hi = None  # no upper bound

        for step in segment.step_results:
            for record in (step.extractor_record, step.auditor_record):
                if record is None:
                    continue
                t = int(record.turn_index)
                if t < turn_lo:
                    continue
                if turn_hi is not None and t > turn_hi:
                    continue
                write_record(out_path, _rebind_record(record, root_session_id=final_sid))

    return out_path
