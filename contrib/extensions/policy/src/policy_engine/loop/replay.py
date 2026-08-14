"""Stage 5: the fitness function, and the only one that means anything.

Fire rate cannot tell a useful check from a noisy one: an item that fires every
time and helps never scores highest. So fitness is counterfactual. Take a
recorded attempt, put the agent back where it was about to finish, inject the
candidate, let it run, and grade the result against what that same attempt
scored without it.

Two arms are run and a third is free:

* ``baseline`` -- what the attempt already scored. Recorded, not re-run.
* ``placebo`` -- a contentless nudge at the same point. Not optional: one extra
  turn of work is worth something on its own, and without this arm an
  improvement cannot be attributed to what the check says.
* ``candidate`` -- the check itself.

**Losses are not zeros.** Of six replays run by hand, two produced a score and
four were lost to infrastructure -- a forked sandbox dies when any sibling
session on the same image idles out and its pool drops the pod. One of those
four had already been shown, from its trajectory, to change the agent's
behaviour exactly as the check intended. Scored as "no improvement" it would
have retired a working check.

How a re-run happens is the benchmark's business; this module only decides what
to run, in what order, and what the numbers mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from .contracts import (
    ARMS,
    CompiledCandidate,
    FailureCase,
    Metrics,
    ReplayMeasurement,
    write_artifact,
)
from .protocols import ReplayBackend

#: What the placebo says: an interruption of the same shape as the candidate and
#: empty of content, so the arm isolates the check's wording from the fact of
#: being stopped and made to look again.
#:
#: Worded for the middle of the work rather than the end, because that is where
#: it now arrives. A placebo that says "before you call this done" would be
#: obviously misplaced at the moment an approach is being chosen, and the
#: comparison would be against a message the agent could dismiss.
#:
#: It ends by handing the work back, and so must anything injected here. An
#: interruption arrives as the newest instruction, and the run stops when the
#: agent answers it without calling a tool -- so a message that can be satisfied
#: by replying ends the attempt where it stands. Measured: two arms that replied
#: once and stopped scored half of what the recording did, not because the
#: message was wrong but because the work was abandoned mid-edit.
PLACEBO = (
    "Quick check-in before you go further -- have another look at where you "
    "have got to, then carry on and finish the task. Nothing is scored yet, "
    "and there is nothing to report back."
)


def replay(
    backend: ReplayBackend,
    compiled: Sequence[CompiledCandidate],
    cases: Sequence[FailureCase],
    *,
    arms: Sequence[str] = ARMS,
    out_path: Path | None = None,
) -> list[ReplayMeasurement]:
    """Every candidate against the case it came from.

    Serial, because two sandboxes for one image race to destroy each other's
    pod. Job directories are named per arm and candidate, so runs for
    *different* cases are safe to overlap, and are.
    """
    by_case = {case.case_id: case for case in cases}
    measurements: list[ReplayMeasurement] = []
    if out_path is not None:
        # Written before the first arm as well as after each one, so the file
        # exists whatever happens next. Callers can then treat this as the sole
        # writer of the artifact, including on the path where nothing is
        # measurable and the loop below never runs.
        write_artifact(out_path, measurements)

    for item in compiled:
        case = by_case.get(item.candidate.from_session)
        if case is None:
            logger.warning(
                "replay: no case {} for candidate {}",
                item.candidate.from_session,
                item.candidate.candidate_id,
            )
            continue
        if not case.replayable:
            logger.warning("replay: {} cannot be re-run; skipping", case.case_id)
            continue

        for arm in arms:
            measurements.append(_measure(backend, case, item, arm))
            if out_path is not None:
                # Written after every arm: a replay run outlasts the window its
                # forks depend on, so partial results must survive a stop.
                write_artifact(out_path, measurements)

    return measurements


def _measure(
    backend: ReplayBackend,
    case: FailureCase,
    item: CompiledCandidate,
    arm: str,
) -> ReplayMeasurement:
    """One arm. Both arms resume at the same turn: comparing a check delivered
    mid-work against a placebo delivered at the end would measure the timing and
    call it the wording."""
    injected = item.candidate.check if arm == "candidate" else PLACEBO
    label = f"replay-{arm}-{item.candidate.candidate_id}"
    resume_at = item.candidate.from_turn
    if resume_at < 0:
        # No decision point was located for this case, so there is nowhere to
        # resume. Reported rather than attempted: the alternative is a re-run
        # from the end of the recording, which answers a question nobody asked
        # and looks like an answer to the one they did.
        logger.warning(
            "replay[{}] {}: no decision point recorded; skipping", arm, case.case_id
        )
        return ReplayMeasurement(
            candidate_id=item.candidate.candidate_id,
            case_id=case.case_id,
            arm=arm,
            outcome="lost",
            lost_reason="no_decision_point",
            requested_turn=resume_at,
            metrics_before=case.metrics,
        )

    logger.info("replay[{}] {} at turn {} -> {}", arm, case.case_id, resume_at, label)
    outcome = backend.rerun(case, injected=injected, label=label, resume_at=resume_at)
    if outcome.lost_reason:
        logger.warning(
            "replay[{}] {}: lost ({})", arm, case.case_id, outcome.lost_reason
        )
        return ReplayMeasurement(
            candidate_id=item.candidate.candidate_id,
            case_id=case.case_id,
            arm=arm,
            outcome="lost",
            lost_reason=outcome.lost_reason,
            requested_turn=resume_at,
            resumed_turn=outcome.resumed_at,
            metrics_before=case.metrics,
            job_dir=outcome.artifact_dir,
        )

    if outcome.resumed_at != resume_at:
        logger.warning(
            "replay[{}] {}: asked to resume at turn {} but resumed at {}; this "
            "row measures a different experiment from the others",
            arm,
            case.case_id,
            resume_at,
            outcome.resumed_at,
        )
    return ReplayMeasurement(
        candidate_id=item.candidate.candidate_id,
        case_id=case.case_id,
        arm=arm,
        outcome=_classify(case, outcome.metrics),
        requested_turn=resume_at,
        resumed_turn=outcome.resumed_at,
        metrics_before=case.metrics,
        metrics_after=outcome.metrics,
        job_dir=outcome.artifact_dir,
    )


def _classify(case: FailureCase, after: Metrics) -> str:
    """Better, worse, or neither.

    Mixed movement is decided by the primary metric, because that is the one the
    benchmark says settles the task. Where a benchmark names no primary, mixed
    movement is not an improvement: something got worse and nothing says the
    trade was worth it.
    """
    delta = after.minus(case.metrics)
    if delta.any_positive and not delta.any_negative:
        return "improved"
    if delta.any_negative and not delta.any_positive:
        return "regressed"
    if delta.any_positive and delta.any_negative:
        primary = delta.get(delta.primary) if delta.primary else None
        if primary is None:
            return "regressed"
        return "improved" if primary > 0 else "regressed"
    return "unchanged"


__all__ = ["PLACEBO", "replay"]
