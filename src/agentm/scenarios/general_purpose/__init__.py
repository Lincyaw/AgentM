"""General-purpose task execution scenario."""


def register() -> None:
    """Register general-purpose types with SDK registries."""
    from agentm.core.state_registry import register_state
    from agentm.core.strategy_registry import register_strategy
    from agentm.models.answer_schemas import ANSWER_SCHEMA

    from agentm.scenarios.general_purpose.answer_schemas import GeneralAnswer
    from agentm.scenarios.general_purpose.state import GeneralPurposeState
    from agentm.scenarios.general_purpose.strategy import GeneralPurposeStrategy

    register_state("general_purpose", GeneralPurposeState)
    register_strategy("general_purpose", GeneralPurposeStrategy())

    ANSWER_SCHEMA.setdefault("execute", GeneralAnswer)
    # No OUTPUT_SCHEMAS entry -- general_purpose has no structured output
