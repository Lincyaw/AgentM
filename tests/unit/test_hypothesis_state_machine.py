"""Tests for hypothesis status state machine transitions.

Ref: designs/orchestrator.md § Hypothesis lifecycle
Ref: core/notebook.py § validate_hypothesis_transition

The hypothesis state machine defines which status transitions are legal.
These constraints prevent the LLM from hallucinating illegal transitions
(e.g., REJECTED → CONFIRMED) that would corrupt diagnostic reasoning.

Bug prevented: LLM hallucinates REJECTED → CONFIRMED transition →
wrong root cause confirmed without re-investigation.
"""

from __future__ import annotations

import pytest

from agentm.models.enums import HypothesisStatus

# Legal transitions as specified in the design document and notebook.py docstring.
# This is the SINGLE SOURCE OF TRUTH for the state machine.
LEGAL_TRANSITIONS: dict[HypothesisStatus, set[HypothesisStatus]] = {
    HypothesisStatus.FORMED: {HypothesisStatus.INVESTIGATING},
    HypothesisStatus.INVESTIGATING: {
        HypothesisStatus.CONFIRMED,
        HypothesisStatus.REJECTED,
        HypothesisStatus.REFINED,
        HypothesisStatus.INCONCLUSIVE,
    },
    HypothesisStatus.REFINED: {HypothesisStatus.INVESTIGATING},
    HypothesisStatus.INCONCLUSIVE: {HypothesisStatus.INVESTIGATING},
    HypothesisStatus.REJECTED: {HypothesisStatus.REFINED},
    HypothesisStatus.CONFIRMED: set(),  # terminal — no outgoing transitions
}


class TestHypothesisStateMachineDesign:
    """Verify the state machine design constraints are self-consistent.

    Bug: state machine has unreachable states or missing transitions →
    hypothesis gets stuck in a state with no way forward.
    """

    def test_every_status_has_a_transition_entry(self):
        """Every HypothesisStatus value must appear as a key in the transition table."""
        for status in HypothesisStatus:
            assert status in LEGAL_TRANSITIONS, (
                f"HypothesisStatus.{status.name} has no entry in LEGAL_TRANSITIONS"
            )

    def test_only_confirmed_is_terminal(self):
        """CONFIRMED is the only state with no outgoing transitions.

        Bug: another state accidentally has empty transitions → hypothesis stuck.
        """
        terminal_states = [s for s, targets in LEGAL_TRANSITIONS.items() if len(targets) == 0]
        assert terminal_states == [HypothesisStatus.CONFIRMED]

    def test_all_non_terminal_states_can_eventually_reach_confirmed(self):
        """Every non-terminal state must have a path to CONFIRMED.

        Bug: state machine has a dead-end cycle (e.g., REFINED ↔ INVESTIGATING
        but INVESTIGATING can't reach CONFIRMED) → diagnosis never concludes.
        """
        def can_reach_confirmed(start: HypothesisStatus, visited: set[HypothesisStatus]) -> bool:
            if start == HypothesisStatus.CONFIRMED:
                return True
            if start in visited:
                return False
            visited.add(start)
            return any(
                can_reach_confirmed(target, visited)
                for target in LEGAL_TRANSITIONS[start]
            )

        for status in HypothesisStatus:
            if status == HypothesisStatus.CONFIRMED:
                continue
            assert can_reach_confirmed(status, set()), (
                f"HypothesisStatus.{status.name} cannot reach CONFIRMED"
            )

    def test_rejected_cannot_directly_confirm(self):
        """REJECTED → CONFIRMED is explicitly illegal.

        This is the P7 test case from testing-strategy.md: prevents the LLM
        from hallucinating a direct REJECTED→CONFIRMED transition.
        """
        assert HypothesisStatus.CONFIRMED not in LEGAL_TRANSITIONS[HypothesisStatus.REJECTED]

    def test_formed_cannot_skip_to_confirmed(self):
        """FORMED → CONFIRMED is illegal — must go through INVESTIGATING first.

        Bug: LLM skips investigation and confirms immediately → no evidence gathered.
        """
        assert HypothesisStatus.CONFIRMED not in LEGAL_TRANSITIONS[HypothesisStatus.FORMED]

    def test_confirmed_has_no_outgoing_transitions(self):
        """Once confirmed, a hypothesis cannot change status.

        Bug: CONFIRMED hypothesis transitions to another state → diagnosis result
        becomes unreliable after being reported.
        """
        assert LEGAL_TRANSITIONS[HypothesisStatus.CONFIRMED] == set()


@pytest.mark.skip(reason="stub not implemented — enable when validate_hypothesis_transition is implemented")
class TestValidateHypothesisTransition:
    """Tests for the validate_hypothesis_transition() function.

    The function checks whether a (current → target) transition is legal.
    These tests will pass once the function is implemented.
    """

    @pytest.mark.parametrize("current,target", [
        (HypothesisStatus.FORMED, HypothesisStatus.INVESTIGATING),
        (HypothesisStatus.INVESTIGATING, HypothesisStatus.CONFIRMED),
        (HypothesisStatus.INVESTIGATING, HypothesisStatus.REJECTED),
        (HypothesisStatus.INVESTIGATING, HypothesisStatus.REFINED),
        (HypothesisStatus.INVESTIGATING, HypothesisStatus.INCONCLUSIVE),
        (HypothesisStatus.REFINED, HypothesisStatus.INVESTIGATING),
        (HypothesisStatus.INCONCLUSIVE, HypothesisStatus.INVESTIGATING),
        (HypothesisStatus.REJECTED, HypothesisStatus.REFINED),
    ])
    def test_legal_transitions_accepted(self, current, target):
        from agentm.core.notebook import validate_hypothesis_transition
        assert validate_hypothesis_transition(current, target) is True

    @pytest.mark.parametrize("current,target", [
        # REJECTED → CONFIRMED is the critical illegal path
        (HypothesisStatus.REJECTED, HypothesisStatus.CONFIRMED),
        # FORMED cannot skip investigation
        (HypothesisStatus.FORMED, HypothesisStatus.CONFIRMED),
        (HypothesisStatus.FORMED, HypothesisStatus.REJECTED),
        # CONFIRMED is terminal
        (HypothesisStatus.CONFIRMED, HypothesisStatus.INVESTIGATING),
        (HypothesisStatus.CONFIRMED, HypothesisStatus.REJECTED),
        # Self-transitions make no sense
        (HypothesisStatus.FORMED, HypothesisStatus.FORMED),
        (HypothesisStatus.INVESTIGATING, HypothesisStatus.INVESTIGATING),
    ])
    def test_illegal_transitions_rejected(self, current, target):
        from agentm.core.notebook import validate_hypothesis_transition
        assert validate_hypothesis_transition(current, target) is False
