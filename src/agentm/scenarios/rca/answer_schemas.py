"""RCA-specific answer schemas for Sub-Agent task types."""

from __future__ import annotations

from pydantic import Field

from agentm.models.answer_schemas import _BaseAnswer


class ScoutAnswer(_BaseAnswer):
    """Scout agent output: structural map of the incident + investigation leads."""

    leads: list[str] = Field(
        description=(
            "3-6 divergent investigation directions. Each lead is one sentence: "
            "'[service/component] may [cause] because [evidence]'. "
            "Cover different fault domains (network, resource, dependency, config, code)."
        ),
    )


class DeepAnalyzeAnswer(_BaseAnswer):
    """Deep-analyze agent output: causal mechanism + refined hypotheses."""

    leads: list[str] = Field(
        description=(
            "1-3 refined hypotheses about specific causal mechanisms. Narrow "
            "and evidence-heavy — only include leads that scout-level analysis "
            "could NOT have produced."
        ),
    )


class VerifyAnswer(_BaseAnswer):
    """Verify agent output: adversarial test verdict with tagged (+)/(-) evidence."""

    verdict: str = Field(
        description=(
            "SUPPORTED, CONTRADICTED, or INCONCLUSIVE — followed by exactly "
            "one sentence citing the strongest piece of evidence. SUPPORTED "
            "means the hypothesis survived active disproof attempts."
        ),
    )


SubAgentAnswer = ScoutAnswer | DeepAnalyzeAnswer | VerifyAnswer
