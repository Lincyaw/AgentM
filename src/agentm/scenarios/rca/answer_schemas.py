"""RCA-specific answer schemas for Sub-Agent task types."""

from __future__ import annotations

from pydantic import Field

from agentm.models.base_answer import _BaseAnswer


class ScoutAnswer(_BaseAnswer):
    """Scout agent output: structural map of the incident."""

    pass


class DeepAnalyzeAnswer(_BaseAnswer):
    """Deep-analyze agent output: causal mechanism explanation."""

    pass


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
