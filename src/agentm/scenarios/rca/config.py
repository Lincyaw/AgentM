"""RCA-specific configuration."""

from __future__ import annotations

from agentm.config.schema import FeatureGatesConfig


class RCAFeatureGates(FeatureGatesConfig):
    """Feature gates specific to hypothesis-driven RCA."""

    adversarial_review: bool = False
    parallel_verification: bool = False
    auto_refine_partial: bool = False
    min_verifications_before_confirm: int = 1
    deep_exploration: bool = False
