"""Pure update-operator data types + shape preconditions.

Design references: §3.3 (operator table), §5.2 (downgrade-application
semantics flip), §7.3 (refine/split/merge gates).

This module is intentionally **not** an atom — no ``MANIFEST``, no
``install``. It is a sibling pure module that the gate atom imports for
shared dataclasses and a small set of structural shape predicates
(``check_refine`` / ``check_split`` / ``check_merge``) that have nothing
to do with semantic judgment of free-text fields.

Phase-2 deletion notes (commit C2 of the LLM-native-judges refactor):

* Removed the Phase-1 ``check_propose`` / ``check_confirm`` /
  ``check_refute`` precondition functions. Their job — deciding whether
  a free-text verdict_proposal "supports" / "triggers" / "steelmans" a
  claim — was free-text judgment masquerading as structural rules. The
  gate now consults the 4 ``rca.judge.*`` services for the same answers.
* Removed all regex helpers (``_TRIGGER_PATTERN`` /
  ``_SUPPORT_PATTERN`` / ``_STEELMAN_PATTERN``) and the
  ``is_prediction_satisfied`` / ``independent_positive_workers`` /
  ``explained_symptom_ids`` helpers they fed. The corresponding
  judgments now live in:

    - ``judge.satisfied``           — was ``is_prediction_satisfied``
    - ``judge.independence``        — was ``independent_positive_workers``
    - ``judge.coverage``            — was ``explained_symptom_ids``
    - ``judge.falsified_genuinely`` — was the "≥1 satisfied negative" rule

* ``UpdateResult.downgraded.applied_id`` is now ``str | None`` and the
  gate returns ``None`` for it: per design §5.2 the gate no longer
  applies the refine itself; the orchestrator decides next steps.

Design decisions worth flagging:

* ``UpdateProposal`` is a **single dataclass with optional payload fields**
  rather than a discriminated union of per-operator dataclasses. The reason
  is twofold: (a) the gate's ``apply`` dispatcher needs a single
  dispatchable type for the public ``rca.gate`` service signature, (b)
  mypy handles the optional-field shape cleanly with ``Optional[...]``
  and runtime ``None`` checks. The free-text ``op`` field is the
  discriminant (per the user-facing spec in ``CLAUDE.md``:
  subjective/classification-shaped fields stay free-text).

* ``UpdateResult`` is a single dataclass keyed by ``kind`` with optional
  variant fields. Same trade-off — easier to consume from the LLM-facing
  tool result layer than three separate classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from agentm_rca_hfsm.schema import (
    CheckResult,
    Hypothesis,
    Observation,
    Symptom,
)


# ---------------------------------------------------------------------------
# Read-side projection of the store the gate uses for cheap structural
# lookups (existing-hypothesis-by-id, refuted branches, ...). Defined as a
# Protocol so ``updates.py`` does not import the store atom; the real
# ``_ReadHandle`` and any test stub both satisfy it structurally.
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

    ``kind`` distinguishes the three variants:

    * ``applied`` populates ``applied_id``.
    * ``downgraded`` populates ``downgrade`` (the operator the gate
      *suggests* the orchestrator run instead) and ``reason``; per design
      §5.2 ``applied_id`` is ``None`` because the gate no longer applies
      the suggested refine itself.
    * ``rejected`` populates ``reason``.

    The semantics flip from Phase 1: ``downgraded.applied_id`` was
    always non-None then (the gate auto-applied a refine of the parent);
    it is ``str | None`` now and gate-emitted values are always ``None``.
    The orchestrator reads ``downgrade`` and ``reason`` and decides
    whether to issue the suggested refine, gather more evidence, split,
    merge, or do something else.
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
        applied_id: str | None = None,
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
# Structural shape predicates. Each returns ``None`` when the shape holds
# and a precise reason string otherwise. These are NOT semantic judgments —
# they are arity / status-pointer checks the gate performs before consulting
# any judge. Semantic decisions live in ``rca.judge.*`` services.
# ---------------------------------------------------------------------------


def check_refine(hypothesis: Hypothesis) -> str | None:
    """§7.3 — refine precondition: the parent must not already be terminal."""

    if hypothesis.status in ("refuted", "confirmed"):
        return f"cannot refine a hypothesis with terminal status {hypothesis.status!r}"
    return None


def check_split(sources: list[Hypothesis], children: list[Hypothesis]) -> str | None:
    """§7.3 — split precondition: exactly one source, ≥2 distinct children."""

    if len(sources) != 1:
        return "split takes exactly one source hypothesis"
    if len(children) < 2:
        return "split must produce at least two children (distinct mechanisms)"
    return None


def check_merge(sources: list[Hypothesis]) -> str | None:
    """§7.3 — merge precondition: ≥2 source hypotheses.

    Phase 1 also computed claim-set overlap on satisfied predictions; that
    overlap test was driven by regex-based "satisfied" semantics and has
    been removed. The merge operator now relies on the orchestrator to
    propose merges only when the structured graph supports them — overlap
    in the LLM-facing sense (do these hypotheses describe the same
    underlying mechanism?) is a judgment, not a structural rule.
    """

    if len(sources) < 2:
        return "merge requires at least two source hypotheses"
    return None


__all__ = [
    "GraphView",
    "UpdateProposal",
    "UpdateResult",
    "ResultKind",
    "check_refine",
    "check_split",
    "check_merge",
]
