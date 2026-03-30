"""Trajectory Judger Scenario implementation using the Scenario protocol.

Minimal wiring scenario for single-pass trajectory classification.
Only requires output_schema - no orchestrator/worker tools needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentm.harness.scenario import ScenarioWiring, SetupContext


class TrajectoryJudgerScenario:
    """Scenario implementation for single-pass trajectory classification.

    Uses a decision-tree methodology to classify RCA agent trajectories
    into categories: success, lucky_hit, exploration_fail, confirmation_fail,
    or judgment_fail.
    """

    @property
    def name(self) -> str:
        return "trajectory_judger"

    def setup(self, _ctx: SetupContext) -> ScenarioWiring:
        """Wire up trajectory judger scenario: only output_schema needed.

        This is a minimal single-pass analysis scenario that requires
        no orchestrator tools, worker tools, or context formatting.
        The harness agent uses output_schema=TrajectoryLabel for
        structured classification output.
        """
        from agentm.harness.scenario import ScenarioWiring
        from agentm.scenarios.trajectory_judger.data import TrajectoryLabel

        return ScenarioWiring(
            output_schema=TrajectoryLabel,
        )
