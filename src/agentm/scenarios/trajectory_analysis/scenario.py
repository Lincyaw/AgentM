"""Trajectory Analysis Scenario implementation using the Scenario protocol.

Wiring: answer schemas (analyze, critique), output schema, and default hooks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentm.harness.scenario import ScenarioWiring, SetupContext


class TrajectoryAnalysisScenario:
    """Scenario implementation for skill-driven trajectory analysis."""

    @property
    def name(self) -> str:
        return "trajectory_analysis"

    def setup(self, ctx: SetupContext) -> ScenarioWiring:
        """Wire up trajectory analysis scenario: schemas and output."""
        from agentm.harness.scenario import ScenarioWiring
        from agentm.scenarios.trajectory_analysis.answer_schemas import (
            AnalyzeAnswer,
            CritiqueAnswer,
        )
        from agentm.scenarios.trajectory_analysis.output import AnalysisReport

        return ScenarioWiring(
            answer_schemas={
                "analyze": AnalyzeAnswer,
                "critique": CritiqueAnswer,
            },
            output_schema=AnalysisReport,
        )
