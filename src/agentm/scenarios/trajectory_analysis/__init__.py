"""Skill-driven trajectory analysis scenario."""


def register() -> None:
    """Register trajectory-analysis scenario with the SDK registry."""
    from agentm.harness.scenario import register_scenario
    from agentm.scenarios.trajectory_analysis.scenario import (
        TrajectoryAnalysisScenario,
    )

    register_scenario(TrajectoryAnalysisScenario())
