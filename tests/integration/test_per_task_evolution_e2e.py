"""Real-LLM E2E experiment fixture for the per-task evolution loop.

**Skipped by default.** Run manually for the parent agent's report:

    ANTHROPIC_API_KEY=... uv run pytest \\
        tests/integration/test_per_task_evolution_e2e.py -v -s

The fixture drives the format_fix tuner with a real LLM provider for up
to 5 iterations and asserts that the tuner activates a tool_normalize_json
version that solves at least 6/8 eval tasks (baseline solves 1/8 — only
task 01_simple_keys passes the naive str.replace).

This is a *budget-burning* test. CI never runs it; presence is for
manual reproducibility of the report.
"""

from __future__ import annotations

import os

import pytest


pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_api_key,
    pytest.mark.skipif(
        not (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("AGENTM_E2E_PROVIDER")
        ),
        reason=(
            "real-LLM E2E requires ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
            "AGENTM_E2E_PROVIDER in env"
        ),
    ),
]


@pytest.mark.asyncio
async def test_format_fix_tuner_evolves_normalize_json() -> None:  # pragma: no cover
    """End-to-end: 5 tuner iterations against the format_fix scenario.
    Asserts post-activation tool_normalize_json passes >= 6/8 eval tasks.

    Skipped unless an API key is configured. The implementation is
    deliberately a placeholder for the parent agent to fill in with the
    concrete provider wiring once a budget is allocated.
    """

    pytest.skip(
        "real-LLM E2E placeholder — wire to the parent agent's chosen "
        "provider (anthropic / openai-compat) and run manually."
    )
