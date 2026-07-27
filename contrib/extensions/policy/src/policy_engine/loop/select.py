"""Stage 6: measurements decide which candidates survive.

Deterministic, and deliberately strict. The rule is not "did the score go up":

* it must go up **on the case it came from**, otherwise the candidate is
  answering a question that case did not pose;
* it must beat its own **placebo** there, otherwise the improvement is the extra
  turn and not the wording;
* it must not **regress** anywhere it was tried, because an item ships to every
  session and one that fixes a task while breaking three is a loss;
* a **lost** measurement is no evidence either way, never a zero. Four of six
  replays run by hand were lost to infrastructure, one of them for a check whose
  trajectory showed it working exactly as intended.

A candidate with no surviving measurement is undecided rather than rejected. The
distinction matters: rejected means measured and not useful, undecided means the
fork window closed before we found out, and the second is worth re-running.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from .contracts import ReplayMeasurement, Verdict


def select(measurements: Sequence[ReplayMeasurement]) -> list[Verdict]:
    by_candidate: dict[str, list[ReplayMeasurement]] = {}
    for measurement in measurements:
        by_candidate.setdefault(measurement.candidate_id, []).append(measurement)

    verdicts = [
        _judge(candidate_id, rows)
        for candidate_id, rows in sorted(by_candidate.items())
    ]
    accepted = sum(1 for v in verdicts if v.accepted)
    logger.info(
        "select: {} accepted, {} not, of {} candidate(s)",
        accepted,
        len(verdicts) - accepted,
        len(verdicts),
    )
    return verdicts


def _judge(candidate_id: str, rows: Sequence[ReplayMeasurement]) -> Verdict:
    candidate_arms = [r for r in rows if r.arm == "candidate"]
    placebo_by_case = {r.case_id: r for r in rows if r.arm == "placebo"}

    improved = tuple(
        sorted(r.case_id for r in candidate_arms if r.outcome == "improved")
    )
    regressed = tuple(
        sorted(r.case_id for r in candidate_arms if r.outcome == "regressed")
    )
    lost = tuple(sorted(r.case_id for r in candidate_arms if r.outcome == "lost"))

    if regressed:
        return Verdict(
            candidate_id=candidate_id,
            accepted=False,
            reason=f"regressed on {len(regressed)} case(s)",
            improved_cases=improved,
            regressed_cases=regressed,
            lost_cases=lost,
        )

    if not improved:
        measured = [r for r in candidate_arms if r.outcome != "lost"]
        if not candidate_arms:
            reason = "no candidate arm was run"
        elif not measured:
            reason = "no measurement survived: every candidate arm was lost"
        else:
            reason = "no case improved"
        return Verdict(
            candidate_id=candidate_id,
            accepted=False,
            reason=reason,
            improved_cases=(),
            regressed_cases=(),
            lost_cases=lost,
        )

    # An improvement the placebo also achieved is the extra turn, not the check.
    beaten = tuple(case for case in improved if _beats_placebo(case, placebo_by_case))
    if not beaten:
        # Why it was not beaten matters, and the two reasons are opposite. A
        # placebo that improved too is a measured refutation. No placebo at all
        # is an unfinished experiment, and saying the first when the second is
        # true reports a finding nobody made.
        unchallenged = [c for c in improved if c not in placebo_by_case]
        reason = (
            f"improved {len(unchallenged)} case(s), but no placebo arm was run "
            "there, so the improvement is unattributed"
            if len(unchallenged) == len(improved)
            else "improved, but its placebo improved as much: the turn, not the wording"
        )
        return Verdict(
            candidate_id=candidate_id,
            accepted=False,
            reason=reason,
            improved_cases=improved,
            lost_cases=lost,
        )

    return Verdict(
        candidate_id=candidate_id,
        accepted=True,
        reason=f"improved {len(beaten)} case(s) beyond placebo",
        improved_cases=beaten,
        lost_cases=lost,
    )


def _beats_placebo(case_id: str, placebo_by_case: dict[str, ReplayMeasurement]) -> bool:
    """No placebo measured means no comparison, and an unchallenged improvement
    is not a demonstrated one."""
    placebo = placebo_by_case.get(case_id)
    if placebo is None or placebo.outcome == "lost":
        return False
    return placebo.outcome != "improved"


__all__ = ["select"]
