"""Layer 4: End-to-end RCA scenario evaluation tests.

Ref: designs/testing-strategy.md -- Layer 4

These tests run complete RCA scenarios and evaluate outcomes.
They are placeholders for future implementation.

Requires: Full system running with LLM API access.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="Layer 4 eval tests not yet implemented")


class TestDatabaseConnectionPoolScenario:
    """End-to-end RCA for a database connection pool exhaustion scenario."""

    def test_identifies_connection_pool_as_root_cause(
        self, rca_scenario_context: dict[str, str]
    ) -> None:
        """The system should identify connection pool exhaustion."""
        raise NotImplementedError("Layer 4 eval not yet implemented")

    def test_exploration_covers_required_data_sources(
        self, rca_scenario_context: dict[str, str]
    ) -> None:
        """Exploration phase should collect data from DB and infra agents."""
        raise NotImplementedError("Layer 4 eval not yet implemented")


class TestHighCPUScenario:
    """End-to-end RCA for a high CPU usage scenario."""

    def test_identifies_cpu_bound_query_as_root_cause(
        self, rca_scenario_context: dict[str, str]
    ) -> None:
        """The system should identify the CPU-intensive query."""
        raise NotImplementedError("Layer 4 eval not yet implemented")
