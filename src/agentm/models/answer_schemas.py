"""Consolidated answer schema registry for all Sub-Agent task types.

Both react/sub_agent.py and node/worker.py import from this module.
Adding a new scenario only requires registering schemas via scenario init.

SDK base class ``_BaseAnswer`` is defined here. Domain-specific schemas
live in their canonical locations under ``scenarios/``.

Registries are populated by scenario ``register()`` functions called
via ``agentm.scenarios.discover()``.  The SDK core never imports from
``scenarios/`` directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared base (SDK)
# ---------------------------------------------------------------------------


class _BaseAnswer(BaseModel):
    """Shared fields across all sub-agent answer types."""

    findings: str = Field(
        description=(
            "Structured findings following the <output> format in your system "
            "prompt. Exact service names in backticks. No reasoning or caveats."
        ),
    )


# ---------------------------------------------------------------------------
# Master registry — imported by both react/sub_agent.py and node/worker.py
# ---------------------------------------------------------------------------

ANSWER_SCHEMA: dict[str, type[BaseModel]] = {}


def get_answer_schema(task_type: str) -> type[BaseModel] | None:
    """Look up an answer schema by task type."""
    return ANSWER_SCHEMA.get(task_type)
