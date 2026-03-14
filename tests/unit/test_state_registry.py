"""Tests for state schema registry.

Ref: designs/generic-state-wrapper.md § State Schema Registry
Ref: designs/system-design-overview.md § Configuration — system.type

The registry maps system_type strings (from scenario.yaml) to TypedDict classes.
AgentSystemBuilder uses this to select the correct state schema at startup.
"""

from __future__ import annotations

import pytest

from agentm.core.state_registry import get_state_schema
from agentm.models.state import (
    DecisionTreeState,
    SequentialDiagnosisState,
)
from agentm.scenarios import discover
from agentm.scenarios.rca.state import HypothesisDrivenState
from agentm.scenarios.memory_extraction.state import MemoryExtractionState

# Ensure scenario registrations are loaded for all tests in this module.
discover()


class TestGetStateSchema:
    """Ref: designs/generic-state-wrapper.md § State Schema Registry — get_state_schema

    Bug: typo in config's system.type → wrong state class silently selected,
    or unhelpful error with no guidance on valid values.
    """

    @pytest.mark.parametrize(
        "system_type,expected",
        [
            ("hypothesis_driven", HypothesisDrivenState),
            ("sequential", SequentialDiagnosisState),
            ("memory_extraction", MemoryExtractionState),
            ("decision_tree", DecisionTreeState),
        ],
    )
    def test_resolves_all_registered_types(self, system_type, expected):
        assert get_state_schema(system_type) is expected

    def test_unknown_type_raises_with_helpful_message(self):
        with pytest.raises(ValueError, match="Unknown system type: foo") as exc_info:
            get_state_schema("foo")
        assert "hypothesis_driven" in str(exc_info.value)

    def test_partial_match_not_accepted(self):
        """Bug: 'hypothesis' matches 'hypothesis_driven' as prefix → wrong type selected."""
        with pytest.raises(ValueError):
            get_state_schema("hypothesis")
