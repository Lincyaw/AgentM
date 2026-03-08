"""Shared fixtures for Layer 3-4 evaluation tests.

These tests use LLM-as-Judge and scenario-level evaluation.
They require external LLM access and are typically run in CI
with appropriate API keys configured.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def rca_scenario_context() -> dict[str, str]:
    """Minimal context for an RCA evaluation scenario."""
    return {
        "task_id": "eval-rca-001",
        "task_description": "Evaluate RCA decision quality",
        "system_type": "hypothesis_driven",
    }
