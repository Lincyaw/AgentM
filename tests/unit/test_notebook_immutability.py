"""Tests for notebook operation immutability contracts.

Ref: designs/orchestrator.md § Data Structures — DiagnosticNotebook
Ref: designs/testing-strategy.md § Layer 1 — Notebook immutability

All notebook operations (add_hypothesis, update_hypothesis_status, etc.)
must return a NEW DiagnosticNotebook instance, leaving the original unchanged.
This is a critical immutability contract — violating it means one graph step's
state mutation leaks into another.

Bug prevented: notebook operation mutates the original → LangGraph checkpoint
captures corrupted state → replay and time-travel produce wrong results.
"""

from __future__ import annotations


from agentm.scenarios.rca.data import DiagnosticNotebook, ExplorationStep
from agentm.scenarios.rca.enums import HypothesisStatus, Phase


def _make_notebook(**overrides) -> DiagnosticNotebook:
    """Create a minimal DiagnosticNotebook for testing."""
    defaults = {
        "task_id": "test-task",
        "task_description": "Test scenario",
        "start_time": "2026-03-08T00:00:00Z",
    }
    defaults.update(overrides)
    return DiagnosticNotebook(**defaults)


class TestAddHypothesisImmutability:
    """add_hypothesis must return a new notebook; original unchanged.

    Bug: in-place dict mutation → hypothesis appears in checkpointed original.
    """

    def test_returns_new_instance(self):
        from agentm.scenarios.rca.notebook import add_hypothesis

        original = _make_notebook()
        result = add_hypothesis(
            original, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z"
        )

        assert result is not original

    def test_original_unchanged(self):
        from agentm.scenarios.rca.notebook import add_hypothesis

        original = _make_notebook()
        original_hyp_count = len(original.hypotheses)

        add_hypothesis(original, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z")

        assert len(original.hypotheses) == original_hyp_count

    def test_new_hypothesis_present_in_result(self):
        from agentm.scenarios.rca.notebook import add_hypothesis

        original = _make_notebook()
        result = add_hypothesis(
            original, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z"
        )

        assert "H1" in result.hypotheses
        assert result.hypotheses["H1"].description == "Pool exhaustion"


class TestUpdateHypothesisStatusImmutability:
    """update_hypothesis_status must return a new notebook; original unchanged.

    Bug: in-place status mutation → rollback via checkpoint doesn't restore
    the previous status.
    """

    def test_returns_new_instance(self):
        from agentm.scenarios.rca.notebook import add_hypothesis, update_hypothesis_status

        nb = _make_notebook()
        nb = add_hypothesis(nb, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z")
        result = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T02:00:00Z"
        )

        assert result is not nb

    def test_original_status_unchanged(self):
        from agentm.scenarios.rca.notebook import add_hypothesis, update_hypothesis_status

        nb = _make_notebook()
        nb = add_hypothesis(nb, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z")

        update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T02:00:00Z"
        )

        assert nb.hypotheses["H1"].status == HypothesisStatus.FORMED


class TestAddExplorationStepImmutability:
    """add_exploration_step must return a new notebook; original unchanged.

    Bug: in-place list append → exploration history shared between checkpoints.
    """

    def test_returns_new_instance(self):
        from agentm.scenarios.rca.notebook import add_exploration_step

        original = _make_notebook()
        step = ExplorationStep(
            step_number=1,
            phase=Phase.EXPLORATION,
            action="dispatch_agent",
            timestamp="2026-03-08T01:00:00Z",
            content="Dispatched db-agent for initial recon",
        )
        result = add_exploration_step(original, step)

        assert result is not original

    def test_original_history_unchanged(self):
        from agentm.scenarios.rca.notebook import add_exploration_step

        original = _make_notebook()
        original_count = len(original.exploration_history)

        step = ExplorationStep(
            step_number=1,
            phase=Phase.EXPLORATION,
            action="dispatch_agent",
            timestamp="2026-03-08T01:00:00Z",
            content="Dispatched db-agent for initial recon",
        )
        add_exploration_step(original, step)

        assert len(original.exploration_history) == original_count


class TestAddCollectedDataImmutability:
    """add_collected_data must return a new notebook; original unchanged.

    Bug: in-place dict mutation → data from one agent's results
    leaks into another checkpoint's view of the notebook.
    """

    def test_returns_new_instance(self):
        from agentm.scenarios.rca.notebook import add_collected_data

        original = _make_notebook()
        result = add_collected_data(original, "db-agent", {"connections": "200/200"})

        assert result is not original

    def test_original_collected_data_unchanged(self):
        from agentm.scenarios.rca.notebook import add_collected_data

        original = _make_notebook()

        add_collected_data(original, "db-agent", {"connections": "200/200"})

        assert "db-agent" not in original.collected_data


class TestSetConfirmedHypothesisImmutability:
    """set_confirmed_hypothesis must return a new notebook; original unchanged.

    Bug: in-place assignment → confirmed_hypothesis set on the original →
    earlier checkpoint sees a confirmed hypothesis prematurely.
    """

    def test_returns_new_instance(self):
        from agentm.scenarios.rca.notebook import (
            add_hypothesis,
            set_confirmed_hypothesis,
            update_hypothesis_status,
        )

        nb = _make_notebook()
        nb = add_hypothesis(nb, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T02:00:00Z"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T03:00:00Z"
        )
        result = set_confirmed_hypothesis(nb, "H1")

        assert result is not nb

    def test_original_confirmed_hypothesis_unchanged(self):
        from agentm.scenarios.rca.notebook import (
            add_hypothesis,
            set_confirmed_hypothesis,
            update_hypothesis_status,
        )

        nb = _make_notebook()
        nb = add_hypothesis(nb, "H1", "Pool exhaustion", "2026-03-08T01:00:00Z")
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.INVESTIGATING, "2026-03-08T02:00:00Z"
        )
        nb = update_hypothesis_status(
            nb, "H1", HypothesisStatus.CONFIRMED, "2026-03-08T03:00:00Z"
        )

        set_confirmed_hypothesis(nb, "H1")

        assert nb.confirmed_hypothesis is None
