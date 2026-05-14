"""Test-only Judge implementations that mimic the Phase-1 structural rules.

These are NOT atoms. They are plain Python classes used by the rca_hfsm
test fixtures to mount stub judges that, given the same structured graph
slice inputs, produce the SAME verdicts the deleted Phase-1
``check_propose`` / ``check_confirm`` / ``check_refute`` /
``is_prediction_satisfied`` / ``independent_positive_workers`` functions
produced. This file is therefore the only place in the post-refactor
repository where the Phase-1 vocabulary survives, and it is intentional:
it documents what the structural rules WERE, so the gate refactor's
behavior-preservation property (design §8.2) can be verified by the
existing fail-stop tests without rewriting them.

Why "mimic" judges instead of hand-scripting each test's
``config.scripted`` list (the C2 directive's option A):

* Hand-scripting is brittle — every test would have to encode the order
  the gate happens to call each judge, and the gate's call sequence is
  itself part of the refactor.
* A reusable mimic class makes the "Phase-1 verdict equivalence"
  property visible in one place. Tests stay unchanged in spirit and
  read the same way.
* These mimics are deliberately structural — they look at typed
  attributes (``polarity``, ``status``, ``worker_session_id``,
  ``related_symptoms``) and a small set of FIXED phrase markers that the
  Phase-1 worker contract guaranteed. They do **not** use regex, and
  they do **not** consult arbitrary free-text; the only string check is
  membership of canonical Phase-1 lemmas in ``verdict_proposal``, which
  is exactly what the deleted regex helpers did. The trade-off:
  this file embeds the Phase-1 vocabulary so the gate file can be free
  of it.

If a future Phase-1 fail-stop test is added with novel inputs whose
Phase-1 verdict differs from these mimics' output, the mimic class
should be extended in lockstep, NOT the gate.
"""

from __future__ import annotations

from typing import Any

from agentm_rca_hfsm.judges import Judge, JudgeContext, Verdict


# Canonical lemma markers the Phase-1 ``verdict_proposal`` field carried.
# These survive in this file only because they were the Phase-1 contract;
# the gate file no longer references them. See design §8.1 (no-regex
# acceptance) for why this separation matters.
_TRIGGER_LEMMAS = ("triggers ", "triggered ", "trigger the", "triggers the", "triggered the")
_SUPPORT_LEMMAS = (
    "support ",
    "supports ",
    "supported ",
    "supports the",
    "supported the",
    "support the",
)
_STEELMAN_LEMMA = "steelman"


def _has_lemma(text: str, lemmas: tuple[str, ...]) -> bool:
    """Token-style lemma check on a free-text field.

    NOT a regex — a tuple of fixed prefixes / phrases that the Phase-1
    worker-return contract guaranteed. The trailing space / "the"
    suffixes in the lemma list are a deliberate dodge of the false
    positives the Phase-1 ``\\b`` word boundaries also dodged
    (e.g. "no triggering observations" should NOT match "triggered").
    """

    lowered = text.lower()
    return any(lemma in lowered for lemma in lemmas)


def _verdict_triggers(check_payload: dict[str, Any]) -> bool:
    return _has_lemma(str(check_payload.get("verdict_proposal", "")), _TRIGGER_LEMMAS)


def _verdict_supports(check_payload: dict[str, Any]) -> bool:
    return _has_lemma(str(check_payload.get("verdict_proposal", "")), _SUPPORT_LEMMAS)


def _is_steelman(check_payload: dict[str, Any]) -> bool:
    return _STEELMAN_LEMMA in str(check_payload.get("verdict_proposal", "")).lower()


def _prediction_satisfied(prediction_payload: dict[str, Any]) -> bool:
    """Phase-1 ``is_prediction_satisfied`` rule, on the slice dict."""

    checks = prediction_payload.get("checks") or []
    if not checks:
        return False
    polarity = prediction_payload.get("polarity")
    if polarity == "negative":
        return not any(_verdict_triggers(c) for c in checks)
    if polarity == "positive":
        return any(_verdict_supports(c) for c in checks)
    return False


class SatisfiedMimic:
    """Mimic of Phase-1 ``is_prediction_satisfied`` semantics."""

    kind = "satisfied"

    def judge(self, context: JudgeContext) -> Verdict:
        prediction = context.graph_slice.get("prediction") or {}
        if _prediction_satisfied(prediction):
            return Verdict(
                verdict="satisfied",
                reason="phase1_mimic: prediction has matching check verdict",
                confidence="high",
            )
        if not prediction.get("checks"):
            return Verdict(
                verdict="unclear",
                reason="phase1_mimic: prediction has no checks",
                confidence="high",
            )
        return Verdict(
            verdict="refuted",
            reason="phase1_mimic: prediction polarity not met by checks",
            confidence="high",
        )


class IndependenceMimic:
    """Mimic of Phase-1 ``worker_session_id`` literal-equality independence.

    Two checks are independent iff their ``worker_session_id`` differs.
    """

    kind = "independence"

    def judge(self, context: JudgeContext) -> Verdict:
        a = context.graph_slice.get("check_a") or {}
        b = context.graph_slice.get("check_b") or {}
        sid_a = str(a.get("worker_session_id", ""))
        sid_b = str(b.get("worker_session_id", ""))
        if sid_a and sid_b and sid_a != sid_b:
            return Verdict(
                verdict="independent",
                reason="phase1_mimic: distinct worker_session_id",
                confidence="high",
            )
        return Verdict(
            verdict="redundant",
            reason=(
                "confirm requires at least two independent worker sessions "
                "producing supporting positive checks (independence requirement); "
                "Phase 1 independence = distinct worker_session_id"
            ),
            confidence="high",
        )


class CoverageMimic:
    """Mimic of Phase-1 chain-walk coverage.

    A symptom is covered iff some satisfied prediction of the candidate
    hypothesis carries an observation that cites the symptom. Phase-1
    folded both the candidate hypothesis' own predictions and any
    already-confirmed hypotheses' predictions into the "explained" set;
    the mimic reproduces the candidate-only half (which is what the
    gate's confirm-time call passes in) plus the candidate's own
    contribution.
    """

    kind = "coverage"

    def judge(self, context: JudgeContext) -> Verdict:
        predictions = context.graph_slice.get("predictions") or []
        symptoms = context.graph_slice.get("symptoms") or []
        covered: set[str] = set()
        for p in predictions:
            if not _prediction_satisfied(p):
                continue
            for c in p.get("checks") or []:
                for o in c.get("observations") or []:
                    covered.update(o.get("related_symptoms") or [])
        unexplained = sorted(
            s.get("id") for s in symptoms if s.get("id") not in covered
        )
        if unexplained:
            return Verdict(
                verdict="gaps",
                reason=(
                    "confirm requires all symptoms to be linked through a satisfied "
                    "prediction of this hypothesis (or another confirmed one); "
                    f"unexplained: {unexplained}"
                ),
                confidence="high",
            )
        return Verdict(
            verdict="covers",
            reason="phase1_mimic: every symptom is linked to a satisfied prediction",
            confidence="high",
        )


class InvestigationGenuineMimic:
    """Mimic of "the orchestrator did something" structural rule.

    Phase-1 had no equivalent — C4 showed the orchestrator could bypass
    the FSM entirely and submit_final_report on an empty trajectory.
    This mimic encodes the simplest structural rule the C5 judge
    replaces with LLM judgment: the trace is genuine iff there is at
    least one symptom AND at least one hypothesis AND at least one
    hypothesis has at least one prediction with at least one check.
    Anything weaker is speculation; the mimic only returns ``unclear``
    on the contradictory shape (no symptoms but mutations were applied).

    Documentation of what we now use LLM judgment to subsume — not
    production code.
    """

    kind = "investigation_genuine"

    def judge(self, context: JudgeContext) -> Verdict:
        gs = context.graph_slice
        symptom_count = int(gs.get("symptom_count", 0) or 0)
        hypotheses = gs.get("hypotheses") or []
        mutations = gs.get("gate_mutations") or {}
        applied = int(mutations.get("applied", 0) or 0)

        # Contradictory: applied mutations but no symptoms. Likely a
        # bus-ordering anomaly — return unclear so the caller treats this
        # as not-genuine without spuriously blaming the LLM.
        if symptom_count == 0 and applied > 0:
            return Verdict(
                verdict="unclear",
                reason=(
                    "phase1_mimic: gate applied mutations but no symptoms "
                    "are recorded — trajectory shape is contradictory"
                ),
                confidence="low",
            )

        if symptom_count == 0:
            return Verdict(
                verdict="speculation",
                reason=(
                    "phase1_mimic: no symptoms recorded — call "
                    "record_symptom for each reported error before "
                    "concluding"
                ),
                confidence="high",
            )
        if not hypotheses:
            return Verdict(
                verdict="speculation",
                reason=(
                    "phase1_mimic: zero hypotheses proposed — propose a "
                    "falsifiable hypothesis and attach checks before "
                    "submitting"
                ),
                confidence="high",
            )
        has_checked_prediction = any(
            int(h.get("checks_count", 0) or 0) > 0 for h in hypotheses
        )
        if not has_checked_prediction:
            return Verdict(
                verdict="speculation",
                reason=(
                    "phase1_mimic: hypotheses proposed but no checks "
                    "attached to their predictions — verify with "
                    "attach_check before submitting"
                ),
                confidence="high",
            )
        return Verdict(
            verdict="genuine_investigation",
            reason=(
                "phase1_mimic: symptoms recorded, hypotheses proposed, "
                "and at least one prediction has been checked"
            ),
            confidence="high",
        )


class FalsifiedGenuinelyMimic:
    """Mimic of Phase-1 falsification / refute structural rules.

    For ``operands["op"] == "refute"`` (refute path): ``genuine_attempt``
    iff EITHER ≥1 negative prediction has a triggered check OR any
    check is marked steelman. This mirrors Phase-1 ``check_refute``.

    For everything else (confirm path): ``genuine_attempt`` iff ≥1
    negative prediction has a SATISFIED check (i.e. a check whose
    verdict does not trigger the claim). Mirrors the Phase-1
    ``_has_satisfied_negative`` rule.
    """

    kind = "falsified_genuinely"

    def judge(self, context: JudgeContext) -> Verdict:
        predictions = context.graph_slice.get("predictions") or []
        op = str(context.operands.get("op", "confirm"))

        if op == "refute":
            for p in predictions:
                if p.get("polarity") == "negative":
                    for c in p.get("checks") or []:
                        if _verdict_triggers(c):
                            return Verdict(
                                verdict="genuine_attempt",
                                reason="phase1_mimic: triggered negative prediction",
                                confidence="high",
                            )
                for c in p.get("checks") or []:
                    if _is_steelman(c):
                        return Verdict(
                            verdict="genuine_attempt",
                            reason="phase1_mimic: steelman check present",
                            confidence="high",
                        )
            return Verdict(
                verdict="no_attempt",
                reason=(
                    "refute requires either a triggered negative prediction or a "
                    "steelman check that failed to find supporting evidence; neither "
                    "is present (use refine if the picture is incomplete)"
                ),
                confidence="high",
            )

        # Confirm path: ≥1 negative prediction must be satisfied.
        for p in predictions:
            if p.get("polarity") != "negative":
                continue
            if _prediction_satisfied(p):
                return Verdict(
                    verdict="genuine_attempt",
                    reason="phase1_mimic: satisfied negative prediction",
                    confidence="high",
                )
        return Verdict(
            verdict="no_attempt",
            reason=(
                "confirm requires at least one satisfied negative prediction; "
                "no negative prediction has a CheckResult whose verdict avoids "
                "triggering the claim (falsification gap)"
            ),
            confidence="high",
        )


def all_mimics() -> dict[str, Judge]:
    """The 4 mimic judges keyed by ``rca.judge.*`` service name."""

    return {
        "rca.judge.satisfied": SatisfiedMimic(),
        "rca.judge.coverage": CoverageMimic(),
        "rca.judge.independence": IndependenceMimic(),
        "rca.judge.falsified_genuinely": FalsifiedGenuinelyMimic(),
        "rca.judge.investigation_genuine": InvestigationGenuineMimic(),
    }


__all__ = [
    "SatisfiedMimic",
    "IndependenceMimic",
    "CoverageMimic",
    "FalsifiedGenuinelyMimic",
    "InvestigationGenuineMimic",
    "all_mimics",
]
