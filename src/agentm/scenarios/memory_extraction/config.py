"""Memory-extraction-specific configuration."""

from __future__ import annotations

from agentm.config.schema import FeatureGatesConfig


class MemoryFeatureGates(FeatureGatesConfig):
    """Feature gates specific to memory extraction."""

    dedup_against_existing: bool = False
    auto_merge_similar: bool = False
    min_evidence_for_pattern: int = 2
