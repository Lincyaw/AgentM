"""Skill-driven trajectory analysis scenario."""


def register() -> None:
    """Register trajectory-analysis types with SDK registries."""
    from agentm.core.state_registry import register_state
    from agentm.core.strategy_registry import register_strategy
    from agentm.models.answer_schemas import ANSWER_SCHEMA
    from agentm.models.output import OUTPUT_SCHEMAS

    from agentm.scenarios.trajectory_analysis.answer_schemas import AnalyzeAnswer
    from agentm.scenarios.trajectory_analysis.output import AnalysisReport
    from agentm.scenarios.trajectory_analysis.state import TrajectoryAnalysisState
    from agentm.scenarios.trajectory_analysis.strategy import TrajectoryAnalysisStrategy

    register_state("trajectory_analysis", TrajectoryAnalysisState)
    register_strategy("trajectory_analysis", TrajectoryAnalysisStrategy())

    ANSWER_SCHEMA.setdefault("analyze", AnalyzeAnswer)
    OUTPUT_SCHEMAS.setdefault("AnalysisReport", AnalysisReport)
