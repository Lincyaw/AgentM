"""Trajectory judger scenario for single-pass trajectory classification.

This scenario analyzes RCA agent trajectories using a decision-tree methodology
to classify success/failure categories:
- success: Correct conclusion with reliable evidence
- lucky_hit: Correct conclusion but problematic evidence
- exploration_fail: Failed to find the root-cause service
- confirmation_fail: Found service but failed to verify it
- judgment_fail: Verified service but reached wrong conclusion
"""
from __future__ import annotations


def register() -> None:
    """Register trajectory-judger scenario with the SDK registry."""
    from agentm.harness.scenario import register_scenario
    from agentm.scenarios.trajectory_judger.scenario import TrajectoryJudgerScenario

    register_scenario(TrajectoryJudgerScenario())
