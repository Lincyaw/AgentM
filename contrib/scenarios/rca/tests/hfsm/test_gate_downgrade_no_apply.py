"""Phase-2 design §5.2 — downgraded confirm does NOT apply the refine.

In Phase 1 a failing ``confirm`` synthesised a refine, applied it, and
left the parent in ``refined→<child>``. That stranded the parent and
broke "attach a second worker, retry confirm" flows. Design §5.2 flips
this semantics: a downgrade returns the suggested refine proposal but
the gate does not mutate the graph; the orchestrator decides next steps.

This file asserts the flip on two axes:

1. After a downgraded confirm, ``result.applied_id is None`` AND the
   parent hypothesis status is still ``"open"`` (NOT
   ``"refined→<child>"``). The suggested refine sits in
   ``result.downgrade`` for the orchestrator to inspect.

2. The same parent can be refined later if the orchestrator chooses to
   apply the refine: an explicit ``UpdateProposal(op="refine", ...)``
   succeeds and the parent moves to ``refined→<child>``. This verifies
   that the downgrade-no-apply path is not blocking the legitimate
   refine path — it's deferring the decision to the orchestrator.
"""

from __future__ import annotations

from agentm_rca.hfsm.schema import Hypothesis, Prediction
from agentm_rca.hfsm.updates import UpdateProposal

from tests.hfsm._gate_fixtures import install_store_and_gate


def _seed_open_hypothesis_with_unchecked_predictions() -> Hypothesis:
    """Build a hypothesis with one negative + one positive prediction.

    Both predictions are unchecked. Under the Phase-1 mimic judges this
    means ``rca.judge.falsified_genuinely`` returns ``"no_attempt"``
    (no satisfied negative), so the confirm path downgrades.
    """

    return Hypothesis(
        id="H-no-apply",
        claim="logrotate failed",
        predictions=[
            Prediction(
                id="P-pos",
                hypothesis_id="H-no-apply",
                claim="logrotate.service is inactive",
                polarity="positive",
            ),
            Prediction(
                id="P-neg",
                hypothesis_id="H-no-apply",
                claim="no rotated archive newer than 7d",
                polarity="negative",
            ),
        ],
    )


def test_downgraded_confirm_does_not_apply_refine() -> None:
    _, gate, read = install_store_and_gate()
    h = _seed_open_hypothesis_with_unchecked_predictions()
    propose_result = gate.apply(UpdateProposal(op="propose", hypothesis=h))
    assert propose_result.kind == "applied", propose_result.reason

    result = gate.apply(UpdateProposal(op="confirm", target_id="H-no-apply"))

    assert result.kind == "downgraded"
    # The semantics-flip: applied_id is None; the suggested refine is in
    # ``downgrade`` but the gate did NOT mutate the graph.
    assert result.applied_id is None
    assert result.downgrade is not None and result.downgrade.op == "refine"
    # Parent hypothesis stays open — NOT stranded in refined→.
    parent = read.get_hypothesis("H-no-apply")
    assert parent is not None
    assert parent.status == "open"
    # No child hypothesis was added.
    assert read.get_hypothesis("H-no-apply.refine") is None


def test_explicit_refine_after_downgrade_still_applies() -> None:
    _, gate, read = install_store_and_gate()
    h = _seed_open_hypothesis_with_unchecked_predictions()
    gate.apply(UpdateProposal(op="propose", hypothesis=h))

    downgrade = gate.apply(UpdateProposal(op="confirm", target_id="H-no-apply"))
    assert downgrade.kind == "downgraded"
    assert read.get_hypothesis("H-no-apply").status == "open"

    # Orchestrator-style: explicitly apply a refine after reading the
    # downgrade's suggestion. The refine is the orchestrator's choice; the
    # gate accepts and the parent finally flips.
    child = Hypothesis(
        id="H-no-apply.refine",
        claim="narrowed claim",
        parent_ids=["H-no-apply"],
        predictions=[
            Prediction(
                id="P-neg-2",
                hypothesis_id="H-no-apply.refine",
                claim="no rotated archive newer than 1d",
                polarity="negative",
            ),
        ],
    )
    refine_result = gate.apply(
        UpdateProposal(op="refine", hypothesis=child, target_id="H-no-apply")
    )
    assert refine_result.kind == "applied", refine_result.reason
    assert read.get_hypothesis("H-no-apply").status == "refined→H-no-apply.refine"
    assert read.get_hypothesis("H-no-apply.refine") is not None
