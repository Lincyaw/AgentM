"""RCA-specific answer schemas for sub-agent task types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class _BaseAnswer(BaseModel):
    findings: str = Field(
        description=(
            "Structured findings following the <output> format in your system "
            "prompt. Exact service names in backticks. No reasoning or caveats."
        )
    )


class ScoutAnswer(_BaseAnswer):
    """Scout agent output: structural map of the incident."""


class DeepAnalyzeAnswer(_BaseAnswer):
    """Deep-analyze agent output: causal mechanism explanation."""


class VerifyAnswer(_BaseAnswer):
    """Verify agent output: adversarial test verdict with tagged evidence."""

    verdict: str = Field(
        description=(
            "SUPPORTED, CONTRADICTED, or INCONCLUSIVE -- followed by exactly "
            "one sentence citing the strongest piece of evidence."
        )
    )


SubAgentAnswer = ScoutAnswer | DeepAnalyzeAnswer | VerifyAnswer


def schema_for_task(task_type: str) -> type[BaseModel]:
    normalized = task_type.strip().lower().replace("-", "_")
    if normalized == "scout":
        return ScoutAnswer
    if normalized == "deep_analyze":
        return DeepAnalyzeAnswer
    if normalized == "verify":
        return VerifyAnswer
    raise ValueError(f"Unknown RCA task type: {task_type!r}")
