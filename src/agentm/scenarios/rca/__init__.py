"""Hypothesis-driven Root Cause Analysis scenario."""


def register() -> None:
    """Register RCA types with SDK registries."""
    from agentm.core.state_registry import register_state
    from agentm.core.strategy_registry import register_strategy
    from agentm.models.answer_schemas import ANSWER_SCHEMA
    from agentm.models.output import OUTPUT_SCHEMAS

    from agentm.scenarios.rca.answer_schemas import (
        DeepAnalyzeAnswer,
        ScoutAnswer,
        VerifyAnswer,
    )
    from agentm.scenarios.rca.output import CausalGraph
    from agentm.scenarios.rca.state import HypothesisDrivenState
    from agentm.scenarios.rca.strategy import HypothesisDrivenStrategy

    register_state("hypothesis_driven", HypothesisDrivenState)
    register_strategy("hypothesis_driven", HypothesisDrivenStrategy())

    ANSWER_SCHEMA.setdefault("scout", ScoutAnswer)
    ANSWER_SCHEMA.setdefault("deep_analyze", DeepAnalyzeAnswer)
    ANSWER_SCHEMA.setdefault("verify", VerifyAnswer)

    OUTPUT_SCHEMAS.setdefault("CausalGraph", CausalGraph)
