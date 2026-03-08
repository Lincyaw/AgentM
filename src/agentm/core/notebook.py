"""DiagnosticNotebook operations (immutable pattern — return new DiagnosticNotebook).

All functions are stubs — raise NotImplementedError.
"""

from __future__ import annotations

from typing import Optional

from agentm.models.data import DiagnosticNotebook, ExplorationStep
from agentm.models.enums import HypothesisStatus, Phase


def add_hypothesis(
    notebook: DiagnosticNotebook,
    hypothesis_id: str,
    description: str,
    created_at: str,
) -> DiagnosticNotebook:
    """Add a new hypothesis to the notebook. Returns a new DiagnosticNotebook."""
    raise NotImplementedError


def update_hypothesis_status(
    notebook: DiagnosticNotebook,
    hypothesis_id: str,
    status: HypothesisStatus,
    last_updated: str,
    evidence: Optional[list[str]] = None,
    counter_evidence: Optional[list[str]] = None,
) -> DiagnosticNotebook:
    """Update a hypothesis's status and optionally its evidence. Returns a new DiagnosticNotebook.

    Legal state transitions (enforced at implementation time):
        formed → investigating
        investigating → confirmed | rejected | refined | inconclusive
        refined → investigating
        inconclusive → investigating
        rejected → refined (only)
        confirmed → (terminal, no outgoing transitions)

    Use validate_hypothesis_transition() to check a transition before applying.
    """
    raise NotImplementedError


def add_exploration_step(
    notebook: DiagnosticNotebook,
    step: ExplorationStep,
) -> DiagnosticNotebook:
    """Add an exploration step to the notebook. Returns a new DiagnosticNotebook."""
    raise NotImplementedError


def add_collected_data(
    notebook: DiagnosticNotebook,
    agent_id: str,
    data: dict,
) -> DiagnosticNotebook:
    """Add collected data from a Sub-Agent. Returns a new DiagnosticNotebook."""
    raise NotImplementedError


def set_confirmed_hypothesis(
    notebook: DiagnosticNotebook,
    hypothesis_id: str,
) -> DiagnosticNotebook:
    """Set the confirmed hypothesis. Returns a new DiagnosticNotebook."""
    raise NotImplementedError


def format_notebook_for_llm(notebook: DiagnosticNotebook) -> str:
    """Format the notebook into an LLM prompt string, using summaries for compressed phases."""
    raise NotImplementedError


def validate_hypothesis_transition(
    current: HypothesisStatus,
    target: HypothesisStatus,
) -> bool:
    """Check whether a hypothesis status transition is valid.

    Legal transitions:
        formed → investigating
        investigating → confirmed | rejected | refined | inconclusive
        refined → investigating
        inconclusive → investigating
        rejected → refined (only)
        confirmed → (terminal, no outgoing transitions)

    Returns True if the transition is allowed, False otherwise.
    """
    raise NotImplementedError
