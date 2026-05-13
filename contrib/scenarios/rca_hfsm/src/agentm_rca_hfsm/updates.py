"""Pure update-operator data types + structural precondition functions.

Design references: §3.3 (operator table), §6.2 (negative-prediction rule),
§7.1 (confirm gate), §7.2 (refute gate), §7.3 (refine/split/merge gates).

This module is intentionally **not** an atom — no ``MANIFEST``, no ``install``.
It is a sibling pure module that the gate atom (and the store atom, for the
"satisfied prediction" helper) imports for shared dataclasses and pure
predicates. Keeping the structural checks here, not inside the gate, makes
them unit-testable without going through the atom install path and lets the
store compute ``get_unexplained_symptoms()`` to the §7.1 definition without
depending on the gate.

Design decisions worth flagging:

* ``UpdateProposal`` is a **single dataclass with optional payload fields**
  rather than a discriminated union of per-operator dataclasses. The reason is
  twofold: (a) the gate's ``apply`` dispatcher needs a single dispatchable
  type for the public ``rca.gate`` service signature, (b) mypy handles the
  optional-field shape cleanly with ``Optional[...]`` and runtime ``None``
  checks; a ``Union[Propose, Confirm, ...]`` would require either
  ``isinstance`` ladders or a ``match`` at every call site. The free-text
  ``op`` field is the discriminant (per the user-facing spec in
  ``CLAUDE.md``: subjective/classification-shaped fields stay free-text).

* ``UpdateResult`` is a single dataclass keyed by ``kind`` with optional
  variant fields. Same trade-off — easier to consume from the LLM-facing
  tool result layer (commit 3) than three separate classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agentm_rca_hfsm.schema import (
    CheckResult,
    Hypothesis,
    Observation,
    Prediction,
    Symptom,
)


# ---------------------------------------------------------------------------
# Read-side projection of the store the gate uses for precondition checks.
# Defined as a Protocol so ``updates.py`` does not import the store atom; the
# real ``_ReadHandle`` and any test stub both satisfy it structurally.
# ---------------------------------------------------------------------------


class GraphView(Protocol):
    def get_symptoms(self) -> list[Symptom]: ...
    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None: ...
    def get_open_leaves(self) -> list[Hypothesis]: ...
    def get_unexplained_symptoms(self) -> list[Symptom]: ...
    def get_refuted_branches(self) -> list[Hypothesis]: ...
    def get_observation_by_signature(
        self, signature: str
    ) -> Observation | None: ...


# ---------------------------------------------------------------------------
# Operator proposal + result types.
# ---------------------------------------------------------------------------


@dataclass
class UpdateProposal:
    """A request to mutate the hypothesis graph.

    The free-text ``op`` field is the discriminant. Per-operator payloads
    populate only the fields they need; the gate validates structurally
    before applying.

    Recognised operator strings (free-text by deliberate choice — see module
    docstring): ``propose``, ``confirm``, ``refute``, ``refine``, ``split``,
    ``merge``, ``supersede``, ``suspend``, ``record_observation``,
    ``attach_check``.
    """

    op: str

    # propose / refine / split / merge / supersede / suspend / confirm / refute
    hypothesis: Hypothesis | None = None
    """Full new hypothesis node for ``propose``/``refine``/``supersede`` and
    the leaves of ``split``. ``confirm``/``refute``/``suspend``/``merge``
    instead reference an existing node via ``target_id``."""

    target_id: str | None = None
    """Identifier of an existing hypothesis the operator acts on
    (``confirm``/``refute``/``suspend``/``supersede``)."""

    children: list[Hypothesis] = field(default_factory=list)
    """Children for ``split``."""

    sources: list[str] = field(default_factory=list)
    """Source hypothesis ids for ``merge``."""

    reason: str = ""
    """Free-text reason — used by ``refine``/``suspend`` and any downgrade."""

    # attach_check
    prediction_id: str | None = None
    check: CheckResult | None = None

    # record_observation
    observation: Observation | None = None

    # symptom intake — not an operator in §3.3 but the gate is the single
    # writer, so it also gates Symptom appends.
    symptom: Symptom | None = None


ResultKind = Literal["applied", "downgraded", "rejected"]


@dataclass
class UpdateResult:
    """Outcome of ``gate.apply``.

    ``kind`` distinguishes the three variants. ``applied`` populates
    ``applied_id``; ``downgraded`` populates ``downgrade`` (the operator the
    gate ran instead) and ``applied_id`` (the id of the node the downgrade
    produced) and ``reason``; ``rejected`` populates ``reason``.

    The downgrade variant carries both the downgraded proposal AND the fact
    that it succeeded — design §7.1: failing preconditions do not raise, they
    downgrade to ``refine`` with a structured reason the LLM can act on.
    """

    kind: ResultKind
    applied_id: str | None = None
    reason: str = ""
    downgrade: UpdateProposal | None = None

    @classmethod
    def applied(cls, node_id: str) -> UpdateResult:
        return cls(kind="applied", applied_id=node_id)

    @classmethod
    def downgraded(
        cls,
        to: UpdateProposal,
        reason: str,
        applied_id: str | None,
    ) -> UpdateResult:
        return cls(
            kind="downgraded",
            applied_id=applied_id,
            reason=reason,
            downgrade=to,
        )

    @classmethod
    def rejected(cls, reason: str) -> UpdateResult:
        return cls(kind="rejected", reason=reason)


# ---------------------------------------------------------------------------
# Structural precondition predicates. Each returns ``None`` when the
# precondition holds and a precise, LLM-actionable reason string otherwise.
# ---------------------------------------------------------------------------


def check_propose(hypothesis: Hypothesis) -> str | None:
    """§6.2 — every freshly proposed ``H`` must declare ≥1 negative prediction.

    The reason string is part of the gate's public contract: tests pin its
    exact wording so the LLM-facing tool result layer (commit 3) can surface
    it verbatim without rewriting.
    """

    if not hypothesis.predictions:
        return "hypothesis must declare at least one negative prediction"
    if not any(p.polarity == "negative" for p in hypothesis.predictions):
        return "hypothesis must declare at least one negative prediction"
    return None


def is_prediction_satisfied(prediction: Prediction) -> bool:
    """A prediction is "satisfied" iff (design §7.1):

    * ≥1 ``CheckResult`` exists for it, AND
    * for ``polarity == "negative"``: no check has reported a verdict whose
      observations triggered the claim, AND
    * for ``polarity == "positive"``: ≥1 check has reported a verdict whose
      observations support the claim.

    Phase 1 "triggered" / "support" are detected by word-boundary regex on
    the worker's free-text ``verdict_proposal`` (CLAUDE.md "no preset enums
    for subjective dimensions" — the worker's verdict stays free-form, the
    gate matches a sentinel word). ``\\btriggered\\b`` and ``\\bsupport(s|ed)?\\b``
    are the canonical Phase 1 markers; "no triggering observations" does
    NOT match because "triggering" is a different lemma. Brief-slice
    disjointness is Phase 2 and is NOT consulted here.
    """

    if not prediction.checks:
        return False
    if prediction.polarity == "negative":
        return not any(_verdict_triggers(c) for c in prediction.checks)
    # positive
    return any(_verdict_supports(c) for c in prediction.checks)


# Word-boundary patterns: deliberately tight so common negations like
# "no triggering observations" or "support not found" are not false-positives.
# The negation half is handled by the worker phrasing its verdict in past-
# perfect / participle form ("supports", "supported", "triggered"); a worker
# returning "no support" or "no triggered observations" simply omits the
# matching lemma.
_TRIGGER_PATTERN = re.compile(r"\b(triggers|triggered)\b", re.IGNORECASE)
_SUPPORT_PATTERN = re.compile(r"\b(support|supports|supported)\b", re.IGNORECASE)
_STEELMAN_PATTERN = re.compile(r"\bsteelman\b", re.IGNORECASE)


def _verdict_triggers(check: CheckResult) -> bool:
    return bool(_TRIGGER_PATTERN.search(check.verdict_proposal))


def _verdict_supports(check: CheckResult) -> bool:
    return bool(_SUPPORT_PATTERN.search(check.verdict_proposal))


def _is_steelman(check: CheckResult) -> bool:
    """Phase 1 steelman signal: free-text marker on the worker's verdict.

    A worker dispatched in steelman mode (commit 4 brief-builder, future)
    embeds the literal ``"steelman"`` token in ``verdict_proposal`` so the
    gate can recognise the attempt without a preset enum.
    """

    return bool(_STEELMAN_PATTERN.search(check.verdict_proposal))


def independent_positive_workers(hypothesis: Hypothesis) -> set[str]:
    """Distinct ``worker_session_id`` set across supporting positive checks.

    Phase 1 independence = different ``worker_session_id``. Brief-slice
    disjointness (design §9 acceptance #9) is Phase 2 and explicitly NOT
    implemented here.
    """

    sessions: set[str] = set()
    for p in hypothesis.predictions:
        if p.polarity != "positive":
            continue
        for c in p.checks:
            if _verdict_supports(c):
                sessions.add(c.worker_session_id)
    return sessions


def _has_satisfied_negative(hypothesis: Hypothesis) -> bool:
    return any(
        p.polarity == "negative" and is_prediction_satisfied(p)
        for p in hypothesis.predictions
    )


def _has_triggered_negative(hypothesis: Hypothesis) -> bool:
    return any(
        p.polarity == "negative"
        and any(_verdict_triggers(c) for c in p.checks)
        for p in hypothesis.predictions
    )


def _has_steelman_check(hypothesis: Hypothesis) -> bool:
    return any(
        any(_is_steelman(c) for c in p.checks)
        for p in hypothesis.predictions
    )


def explained_symptom_ids(graph: GraphView) -> set[str]:
    """Symptoms linked through a satisfied prediction of a confirmed hypothesis.

    Implementation of the §7.1 "covers all unexplained symptoms" half. A
    symptom is explained iff there exists an ``Observation`` whose
    ``related_symptoms`` contains it AND that observation appears on a
    ``CheckResult`` of a satisfied prediction of a ``confirmed`` hypothesis.

    Confirmed-hypothesis enumeration is delegated to a graph-side helper
    via ``get_confirmed`` when the read view exposes it (the store does);
    when absent, no hypothesis is considered confirmed and every cited
    symptom remains unexplained.
    """

    explained: set[str] = set()
    for h in _confirmed_hypotheses(graph):
        for p in h.predictions:
            if not is_prediction_satisfied(p):
                continue
            for c in p.checks:
                for obs in c.observations:
                    explained.update(obs.related_symptoms)
    return explained


def _confirmed_hypotheses(graph: GraphView) -> list[Hypothesis]:
    """Best-effort enumeration of confirmed hypotheses on a ``GraphView``.

    Calls ``get_confirmed`` if the view exposes it (the store does, see
    commit 2 store tightening); otherwise returns an empty list. This keeps
    the ``GraphView`` Protocol minimal while allowing the store to extend.
    """

    getter = getattr(graph, "get_confirmed", None)
    if callable(getter):
        result = getter()
        if isinstance(result, list):
            return result
    return []


def check_confirm(hypothesis: Hypothesis, graph: GraphView) -> str | None:
    """§7.1 — confirm gate.

    Returns ``None`` if all three preconditions hold; otherwise a precise
    reason string identifying the missing piece so the gate can downgrade to
    ``refine(H, reason)``.
    """

    if not _has_satisfied_negative(hypothesis):
        return (
            "confirm requires at least one satisfied negative prediction; "
            "no negative prediction has a CheckResult whose verdict avoids "
            "triggering the claim (falsification gap)"
        )
    if len(independent_positive_workers(hypothesis)) < 2:
        return (
            "confirm requires at least two independent worker sessions "
            "producing supporting positive checks (independence requirement); "
            "Phase 1 independence = distinct worker_session_id"
        )
    unexplained = [
        s.id for s in graph.get_symptoms()
        if s.id not in explained_symptom_ids(graph)
        # NOTE: at apply-time the hypothesis is still ``open``; its own
        # satisfied predictions are not yet in the "explained" set. We
        # therefore unconditionally fold this hypothesis' satisfied
        # predictions in too so the precondition is "the confirmation would
        # cover all symptoms" rather than "all symptoms were already covered
        # by *other* confirmed hypotheses".
        and s.id not in _symptoms_covered_by(hypothesis)
    ]
    if unexplained:
        return (
            "confirm requires all symptoms to be linked through a satisfied "
            "prediction of this hypothesis (or another confirmed one); "
            f"unexplained: {sorted(unexplained)}"
        )
    return None


def _symptoms_covered_by(hypothesis: Hypothesis) -> set[str]:
    covered: set[str] = set()
    for p in hypothesis.predictions:
        if not is_prediction_satisfied(p):
            continue
        for c in p.checks:
            for obs in c.observations:
                covered.update(obs.related_symptoms)
    return covered


def check_refute(hypothesis: Hypothesis, graph: GraphView) -> str | None:
    """§7.2 — refute gate.

    Refutation is accepted on EITHER of two structural grounds (asymmetric
    on purpose, per design): a triggered negative prediction (Popperian
    falsification), OR a steelman check that itself failed to find support.
    """

    del graph  # refute checks are local to the hypothesis in Phase 1
    if _has_triggered_negative(hypothesis):
        return None
    if _has_steelman_check(hypothesis):
        return None
    return (
        "refute requires either a triggered negative prediction or a "
        "steelman check that failed to find supporting evidence; neither "
        "is present (use refine if the picture is incomplete)"
    )


def check_refine(hypothesis: Hypothesis) -> str | None:
    """§7.3 — refine precondition. Phase 1 keeps this light: the parent must
    exist and not already be terminal (``refuted`` / ``confirmed``).
    """

    if hypothesis.status in ("refuted", "confirmed"):
        return f"cannot refine a hypothesis with terminal status {hypothesis.status!r}"
    return None


def check_split(sources: list[Hypothesis], children: list[Hypothesis]) -> str | None:
    """§7.3 — split precondition: ≥2 distinct mechanisms enumerated as children."""

    if len(sources) != 1:
        return "split takes exactly one source hypothesis"
    if len(children) < 2:
        return "split must produce at least two children (distinct mechanisms)"
    return None


def check_merge(sources: list[Hypothesis]) -> str | None:
    """§7.3 — merge precondition: ≥2 sources with overlapping satisfied predictions.

    Overlap here is the symmetric-difference test on satisfied-prediction
    claims. The gate (not the worker) computes this — workers cannot
    self-declare overlap.
    """

    if len(sources) < 2:
        return "merge requires at least two source hypotheses"
    claim_sets = [
        {p.claim for p in h.predictions if is_prediction_satisfied(p)}
        for h in sources
    ]
    # Pairwise overlap among all sources.
    base = claim_sets[0]
    for other in claim_sets[1:]:
        if not (base & other):
            return (
                "merge requires overlap in satisfied prediction claims "
                "across all source hypotheses"
            )
    return None


__all__ = [
    "GraphView",
    "UpdateProposal",
    "UpdateResult",
    "ResultKind",
    "check_propose",
    "check_confirm",
    "check_refute",
    "check_refine",
    "check_split",
    "check_merge",
    "is_prediction_satisfied",
    "independent_positive_workers",
    "explained_symptom_ids",
]
