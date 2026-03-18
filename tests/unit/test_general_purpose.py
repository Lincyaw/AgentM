"""Tests for the general_purpose scenario: state, strategy, answer schemas,
registration, ScenarioToolBundle, and builder integration.

Each test prevents a specific class of bugs:
- Strategy protocol compliance → missing methods → crash at graph execution
- State schema correctness → wrong field types → LangGraph reducer errors
- Registration → missing types → builder lookup crash
- ScenarioToolBundle → strategy returns correct (empty) bundle → builder wiring
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.strategy import ReasoningStrategy, get_scenario_tools
from agentm.scenarios import discover

# Ensure scenario registrations are loaded.
discover()


# ---------------------------------------------------------------------------
# Strategy Protocol
# ---------------------------------------------------------------------------


class TestGeneralPurposeStrategyProtocol:
    """GeneralPurposeStrategy satisfies ReasoningStrategy protocol."""

    def test_satisfies_protocol(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        strategy = GeneralPurposeStrategy()
        assert isinstance(strategy, ReasoningStrategy)

    def test_name(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        assert GeneralPurposeStrategy().name == "general_purpose"

    def test_initial_state_has_required_fields(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        strategy = GeneralPurposeStrategy()
        state = strategy.initial_state("t-1", "do something")

        assert state["task_id"] == "t-1"
        assert state["task_description"] == "do something"
        assert state["current_phase"] == "execute"
        assert state["active_skills"] == []
        assert state["skill_cache"] == {}
        assert state["conversation_facts"] == []
        assert state["structured_response"] is None
        assert len(state["messages"]) == 1

    def test_should_terminate_always_false(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        strategy = GeneralPurposeStrategy()
        state = strategy.initial_state("t-1", "test")
        assert strategy.should_terminate(state) is False

    def test_single_self_looping_phase(self):
        """Bug prevented: missing or incorrect phase definitions → PhaseManager crashes."""
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        phases = GeneralPurposeStrategy().phase_definitions()
        assert "execute" in phases
        assert len(phases) == 1
        assert "execute" in phases["execute"].next_phases

    def test_no_output_schema(self):
        """Bug prevented: structured output schema on GP → synthesize node fails."""
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        assert GeneralPurposeStrategy().get_output_schema() is None

    def test_answer_schemas_has_execute(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        schemas = GeneralPurposeStrategy().get_answer_schemas()
        assert "execute" in schemas

    def test_state_schema_returns_correct_type(self):
        from agentm.scenarios.general_purpose.state import GeneralPurposeState
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        assert GeneralPurposeStrategy().state_schema() is GeneralPurposeState


# ---------------------------------------------------------------------------
# State Immutability
# ---------------------------------------------------------------------------


class TestGeneralPurposeStateImmutability:
    """compress_state returns new dict; original is unchanged."""

    def test_compress_state_no_mutation(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        strategy = GeneralPurposeStrategy()
        state = strategy.initial_state("t-1", "test")

        # Inject 60 facts to trigger trimming
        facts = [f"fact-{i}" for i in range(60)]
        state_with_facts = {**state, "conversation_facts": facts}

        compressed = strategy.compress_state(state_with_facts, "execute")

        # Original unchanged
        assert len(state_with_facts["conversation_facts"]) == 60
        # Compressed has only last 30
        assert len(compressed["conversation_facts"]) == 30
        assert compressed["conversation_facts"][-1] == "fact-59"

    def test_compress_state_under_threshold_is_noop(self):
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        strategy = GeneralPurposeStrategy()
        state = strategy.initial_state("t-1", "test")
        facts = [f"fact-{i}" for i in range(10)]
        state_with_facts = {**state, "conversation_facts": facts}

        compressed = strategy.compress_state(state_with_facts, "execute")
        assert compressed is state_with_facts  # Same object — no copy needed


# ---------------------------------------------------------------------------
# Formatter (inlined into strategy.format_context)
# ---------------------------------------------------------------------------


class TestGeneralPurposeFormatter:
    """strategy.format_context produces correct text for system prompt."""

    def _format(self, state: dict[str, Any]) -> str:
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        return GeneralPurposeStrategy().format_context(state)

    def test_empty_state(self):
        state: dict[str, Any] = {
            "active_skills": [],
            "skill_cache": {},
            "conversation_facts": [],
        }
        result = self._format(state)
        assert "no context yet" in result.lower()

    def test_skill_injection(self):
        state: dict[str, Any] = {
            "active_skills": ["skill/data-analysis"],
            "skill_cache": {"skill/data-analysis": "Use pandas for analysis."},
            "conversation_facts": [],
        }
        result = self._format(state)
        assert "Active Skills" in result
        assert "skill/data-analysis" in result
        assert "Use pandas for analysis." in result

    def test_skill_context_cap(self):
        """Bug prevented: unbounded skill context → prompt token overflow."""
        large_body = "x" * 10000
        state: dict[str, Any] = {
            "active_skills": ["skill/big"],
            "skill_cache": {"skill/big": large_body},
            "conversation_facts": [],
        }
        result = self._format(state)
        # Body should be truncated to 8000 chars
        assert len(result) < len(large_body)

    def test_facts_injection(self):
        facts = [f"fact-{i}" for i in range(25)]
        state: dict[str, Any] = {
            "active_skills": [],
            "skill_cache": {},
            "conversation_facts": facts,
        }
        result = self._format(state)
        assert "Conversation Facts" in result
        assert "25 total" in result
        # Only last 20 displayed
        assert "fact-24" in result
        assert "5 earlier facts omitted" in result


# ---------------------------------------------------------------------------
# Answer Schema
# ---------------------------------------------------------------------------


class TestGeneralAnswer:
    """GeneralAnswer schema for worker structured output."""

    def test_answer_field(self):
        from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer

        answer = GeneralAnswer(answer="The service is healthy.")
        assert answer.answer == "The service is healthy."

    def test_answer_required(self):
        from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GeneralAnswer()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Scenario registration populates SDK registries correctly."""

    def test_state_registry(self):
        from agentm.core.state_registry import get_state_schema
        from agentm.scenarios.general_purpose.state import GeneralPurposeState

        assert get_state_schema("general_purpose") is GeneralPurposeState

    def test_strategy_registry(self):
        from agentm.core.strategy_registry import get_strategy

        strategy = get_strategy("general_purpose")
        assert strategy.name == "general_purpose"

    def test_answer_schema_registry(self):
        from agentm.models.answer_schemas import ANSWER_SCHEMA

        assert "execute" in ANSWER_SCHEMA

    def test_no_collision_with_existing_scenarios(self):
        """Bug prevented: overwriting RCA's answer schemas → wrong worker output."""
        from agentm.models.answer_schemas import ANSWER_SCHEMA

        assert "scout" in ANSWER_SCHEMA
        assert "deep_analyze" in ANSWER_SCHEMA
        assert "verify" in ANSWER_SCHEMA


# ---------------------------------------------------------------------------
# ScenarioToolBundle (strategy-based tool injection)
# ---------------------------------------------------------------------------


class TestScenarioToolBundle:
    """Strategy.create_scenario_tools returns correct bundles."""

    def test_gp_strategy_returns_empty_bundle(self, tmp_path):
        """GP strategy returns empty bundle — skill access is via vault tools."""
        from agentm.tools.vault.store import MarkdownVault
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        vault = MarkdownVault(tmp_path / "vault")
        strategy = GeneralPurposeStrategy()
        bundle = strategy.create_scenario_tools(vault=vault)

        assert bundle.orchestrator_tools == {}
        assert bundle.format_context_override is None
        assert bundle.worker_tools == []

    def test_gp_strategy_no_vault_returns_empty(self):
        """No vault → empty bundle (graceful degradation)."""
        from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

        bundle = GeneralPurposeStrategy().create_scenario_tools()
        assert bundle.orchestrator_tools == {}

    def test_rca_strategy_returns_rca_tools(self):
        """RCA strategy also uses create_scenario_tools (no builder branching)."""
        from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

        strategy = HypothesisDrivenStrategy()
        bundle = strategy.create_scenario_tools(trajectory=None)

        assert "update_hypothesis" in bundle.orchestrator_tools
        assert bundle.format_context_override is not None
        assert len(bundle.worker_tools) > 0

    def test_get_scenario_tools_fallback(self):
        """Strategy without create_scenario_tools returns empty bundle."""
        from agentm.models.data import ScenarioToolBundle
        from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

        strategy = MemoryExtractionStrategy()
        bundle = get_scenario_tools(strategy)

        assert bundle.orchestrator_tools == {}
        assert bundle.worker_tools == []
        assert bundle.format_context_override is None


# ---------------------------------------------------------------------------
# Builder Integration
# ---------------------------------------------------------------------------


class TestBuilderIntegration:
    """Builder uses strategy protocol instead of if/else chains."""

    def test_no_resolve_format_context_function(self):
        """Bug prevented: _resolve_format_context still exists → pluggability broken."""
        import agentm.builder as builder_module

        assert not hasattr(builder_module, "_resolve_format_context")

    def test_strategy_format_context_used_for_all_scenarios(self):
        """All registered strategies provide format_context via protocol."""
        from agentm.core.strategy_registry import get_strategy

        for system_type in ("hypothesis_driven", "memory_extraction", "general_purpose"):
            strategy = get_strategy(system_type)
            assert callable(strategy.format_context)
