"""P7, P8: Hypothesis lifecycle tests — illegal transitions and consistency.

Bug prevented:
- P7: LLM hallucinates REJECTED→CONFIRMED → wrong root cause confirmed
- P8: confirmed_hypothesis set for non-CONFIRMED hypothesis → inconsistent Notebook
"""

from __future__ import annotations

import pytest

from agentm.scenarios.rca.notebook import (
    add_hypothesis,
    set_confirmed_hypothesis,
    update_hypothesis_status,
)
from agentm.scenarios.rca.data import DiagnosticNotebook
from agentm.scenarios.rca.enums import HypothesisStatus


class TestUpdateHypothesisRejectsIllegalTransitions:
    """P7: update_hypothesis rejects illegal state transitions."""

    def test_rejected_to_confirmed_raises(self, notebook: DiagnosticNotebook) -> None:
        """REJECTED → CONFIRMED is illegal — must go through REFINED first."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:01:00"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.REJECTED, "2026-03-08T10:02:00"
        )

        with pytest.raises(ValueError, match="Illegal hypothesis transition"):
            update_hypothesis_status(
                nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T10:03:00"
            )

    def test_formed_to_confirmed_raises(self, notebook: DiagnosticNotebook) -> None:
        """FORMED → CONFIRMED is illegal — must go through INVESTIGATING."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")

        with pytest.raises(ValueError, match="Illegal hypothesis transition"):
            update_hypothesis_status(
                nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T10:01:00"
            )

    def test_confirmed_to_anything_raises(self, notebook: DiagnosticNotebook) -> None:
        """CONFIRMED is terminal — no outgoing transitions."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:01:00"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T10:02:00"
        )

        with pytest.raises(ValueError, match="Illegal hypothesis transition"):
            update_hypothesis_status(
                nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:03:00"
            )

    def test_notebook_unchanged_after_rejected_transition(
        self, notebook: DiagnosticNotebook
    ) -> None:
        """After a rejected transition, the notebook must NOT be modified."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:01:00"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.REJECTED, "2026-03-08T10:02:00"
        )

        original_status = nb.hypotheses["H1"].status

        try:
            update_hypothesis_status(
                nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T10:03:00"
            )
        except ValueError:
            pass

        assert nb.hypotheses["H1"].status == original_status

    def test_legal_transition_rejected_to_refined(
        self, notebook: DiagnosticNotebook
    ) -> None:
        """REJECTED → REFINED is the only legal path from REJECTED."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:01:00"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.REJECTED, "2026-03-08T10:02:00"
        )

        nb2 = update_hypothesis_status(
            nb, "H1", HypothesisStatus.REFINED, "2026-03-08T10:03:00"
        )
        assert nb2.hypotheses["H1"].status == HypothesisStatus.REFINED


class TestSetConfirmedHypothesisConsistency:
    """P8: set_confirmed_hypothesis should only confirm CONFIRMED hypotheses."""

    def test_confirm_investigating_hypothesis_raises(
        self, notebook_with_hypothesis: DiagnosticNotebook
    ) -> None:
        """Setting confirmed_hypothesis for an INVESTIGATING hypothesis raises ValueError.

        The function validates that the hypothesis must have CONFIRMED status
        before it can be set as the confirmed root cause.
        """
        # notebook_with_hypothesis has H1 in INVESTIGATING state
        with pytest.raises(ValueError, match="expected 'confirmed'"):
            set_confirmed_hypothesis(notebook_with_hypothesis, "H1")

    def test_confirm_after_proper_lifecycle(self, notebook: DiagnosticNotebook) -> None:
        """Proper lifecycle: FORMED → INVESTIGATING → CONFIRMED → set_confirmed."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:01:00"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T10:02:00"
        )
        nb = set_confirmed_hypothesis(nb, "H1")

        assert nb.confirmed_hypothesis == "H1"
        assert nb.hypotheses["H1"].status == HypothesisStatus.CONFIRMED

    def test_set_confirmed_is_immutable(self, notebook: DiagnosticNotebook) -> None:
        """set_confirmed_hypothesis should not modify the original notebook."""
        nb = add_hypothesis(notebook, "H1", "Pool exhaustion", "2026-03-08T10:00:00")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T10:01:00"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T10:02:00"
        )
        original = nb
        result = set_confirmed_hypothesis(original, "H1")

        assert result is not original
        assert original.confirmed_hypothesis is None
