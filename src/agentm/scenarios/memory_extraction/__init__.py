"""Cross-task knowledge extraction scenario."""


def register() -> None:
    """Register memory-extraction types with SDK registries."""
    from agentm.core.state_registry import register_state
    from agentm.core.strategy_registry import register_strategy
    from agentm.models.answer_schemas import ANSWER_SCHEMA
    from agentm.models.output import OUTPUT_SCHEMAS

    from agentm.scenarios.memory_extraction.answer_schemas import (
        AnalyzeAnswer,
        CollectAnswer,
        ExtractAnswer,
        RefineAnswer,
    )
    from agentm.scenarios.memory_extraction.output import KnowledgeSummary
    from agentm.scenarios.memory_extraction.state import MemoryExtractionState
    from agentm.scenarios.memory_extraction.strategy import MemoryExtractionStrategy

    register_state("memory_extraction", MemoryExtractionState)
    register_strategy("memory_extraction", MemoryExtractionStrategy())

    ANSWER_SCHEMA.setdefault("collect", CollectAnswer)
    ANSWER_SCHEMA.setdefault("analyze", AnalyzeAnswer)
    ANSWER_SCHEMA.setdefault("extract", ExtractAnswer)
    ANSWER_SCHEMA.setdefault("refine", RefineAnswer)

    OUTPUT_SCHEMAS.setdefault("KnowledgeSummary", KnowledgeSummary)
