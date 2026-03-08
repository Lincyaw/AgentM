"""Tests for design-document contract compliance.

These tests verify that code data structures match the design documents'
normative definitions. They prevent drift between design and implementation.
"""

from __future__ import annotations

from agentm.models.data import DiagnosticNotebook, Hypothesis
from agentm.models.enums import HypothesisStatus, Phase, Verdict


class TestHypothesisStatusMatchesDesign:
    """Ref: designs/orchestrator.md § Orchestrator Tools — update_hypothesis

    The update_hypothesis tool declares:
        status: Literal["formed", "investigating", "confirmed", "rejected", "refined", "inconclusive"]
    The HypothesisStatus enum must have exactly these values.

    Bug: enum drifts from tool signature → LLM produces a status the code rejects.
    """

    DESIGN_VALUES = {"formed", "investigating", "confirmed", "rejected", "refined", "inconclusive"}

    def test_enum_values_match_design_document(self):
        actual = {member.value for member in HypothesisStatus}
        assert actual == self.DESIGN_VALUES, (
            f"HypothesisStatus drift detected.\n"
            f"  Missing from code: {self.DESIGN_VALUES - actual}\n"
            f"  Extra in code: {actual - self.DESIGN_VALUES}"
        )


class TestVerdictMatchesDesign:
    """Ref: designs/orchestrator.md § Data Structures — VerificationResult

    VerificationResult.verdict is a three-value enum: confirmed | rejected | partial.

    Bug: verdict values drift → orchestrator routing logic breaks on unexpected strings.
    """

    DESIGN_VALUES = {"confirmed", "rejected", "partial"}

    def test_enum_values_match_design_document(self):
        actual = {member.value for member in Verdict}
        assert actual == self.DESIGN_VALUES


class TestPhaseMatchesDesign:
    """Ref: designs/orchestrator.md § Overview — Four Diagnostic Phases

    Phases are Notebook state markers: exploration, generation, verification, confirmation.

    Bug: missing phase → Notebook current_phase set to invalid value.
    """

    DESIGN_VALUES = {"exploration", "generation", "verification", "confirmation"}

    def test_enum_values_match_design_document(self):
        actual = {member.value for member in Phase}
        assert actual == self.DESIGN_VALUES


class TestDiagnosticNotebookIsolation:
    """Ref: designs/orchestrator.md § Data Structures — DiagnosticNotebook

    DiagnosticNotebook uses mutable container fields (dict, list).
    Each instance must have independent containers (via default_factory).

    Bug: shared mutable default → hypothesis added to one Notebook appears in another.
    """

    def test_hypotheses_dict_independent_between_instances(self):
        nb1 = DiagnosticNotebook(task_id="t1", task_description="d", start_time="s")
        nb2 = DiagnosticNotebook(task_id="t2", task_description="d", start_time="s")
        nb1.hypotheses["h1"] = Hypothesis(id="h1", description="test")
        assert "h1" not in nb2.hypotheses

    def test_exploration_history_independent_between_instances(self):
        nb1 = DiagnosticNotebook(task_id="t1", task_description="d", start_time="s")
        nb2 = DiagnosticNotebook(task_id="t2", task_description="d", start_time="s")
        nb1.exploration_history.append(object())
        assert len(nb2.exploration_history) == 0

    def test_collected_data_independent_between_instances(self):
        nb1 = DiagnosticNotebook(task_id="t1", task_description="d", start_time="s")
        nb2 = DiagnosticNotebook(task_id="t2", task_description="d", start_time="s")
        nb1.collected_data["agent-1"] = {"cpu": 90}
        assert nb2.collected_data == {}


class TestExecutorStateContract:
    """Ref: designs/orchestrator.md § Data Structures — ExecutorState

    ExecutorState (TypedDict) must contain all fields the orchestrator design requires.

    Bug: missing field → create_react_agent state schema incomplete → runtime KeyError.
    """

    def test_executor_state_has_all_required_fields(self):
        from agentm.models.state import ExecutorState
        required = ("messages", "notebook", "task_id", "current_phase", "compression_refs")
        annotations = ExecutorState.__annotations__
        for field in required:
            assert field in annotations, f"ExecutorState missing required field: {field}"


class TestSubAgentStateContract:
    """Ref: designs/sub-agent.md § State Schema — SubAgentState

    SubAgentState must contain all fields the sub-agent design requires.

    Bug: missing field → sub-agent subgraph cannot track tool calls or compression.
    """

    def test_sub_agent_state_has_all_required_fields(self):
        from agentm.models.state import SubAgentState
        required = ("messages", "scratchpad", "observations", "tool_call_count", "compression_refs")
        annotations = SubAgentState.__annotations__
        for field in required:
            assert field in annotations, f"SubAgentState missing required field: {field}"
