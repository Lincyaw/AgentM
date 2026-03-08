"""DiagnosticNotebook operations (immutable pattern — return new DiagnosticNotebook).

All functions return a new DiagnosticNotebook instance. The original is never modified.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from agentm.models.data import DiagnosticNotebook, ExplorationStep, Hypothesis
from agentm.models.enums import HypothesisStatus, Phase

_LEGAL_TRANSITIONS: dict[HypothesisStatus, set[HypothesisStatus]] = {
    HypothesisStatus.FORMED: {HypothesisStatus.INVESTIGATING},
    HypothesisStatus.INVESTIGATING: {
        HypothesisStatus.CONFIRMED,
        HypothesisStatus.REJECTED,
        HypothesisStatus.REFINED,
        HypothesisStatus.INCONCLUSIVE,
    },
    HypothesisStatus.REFINED: {HypothesisStatus.INVESTIGATING},
    HypothesisStatus.INCONCLUSIVE: {HypothesisStatus.INVESTIGATING},
    HypothesisStatus.REJECTED: {HypothesisStatus.REFINED},
    HypothesisStatus.CONFIRMED: set(),
}


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
    if current == target:
        return False
    return target in _LEGAL_TRANSITIONS.get(current, set())


def add_hypothesis(
    notebook: DiagnosticNotebook,
    hypothesis_id: str,
    description: str,
    created_at: str,
) -> DiagnosticNotebook:
    """Add a new hypothesis to the notebook. Returns a new DiagnosticNotebook."""
    new_hypotheses = {**notebook.hypotheses}
    new_hypotheses[hypothesis_id] = Hypothesis(
        id=hypothesis_id,
        description=description,
        created_at=created_at,
    )
    return replace(notebook, hypotheses=new_hypotheses)


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
    existing = notebook.hypotheses[hypothesis_id]
    if not validate_hypothesis_transition(existing.status, status):
        raise ValueError(
            f"Illegal hypothesis transition: {existing.status} → {status} "
            f"(hypothesis_id={hypothesis_id!r})"
        )

    new_evidence = list(existing.evidence) + (evidence or [])
    new_counter_evidence = list(existing.counter_evidence) + (counter_evidence or [])

    updated = Hypothesis(
        id=existing.id,
        description=existing.description,
        evidence=new_evidence,
        counter_evidence=new_counter_evidence,
        status=status,
        created_at=existing.created_at,
        last_updated=last_updated,
    )

    new_hypotheses = {**notebook.hypotheses, hypothesis_id: updated}
    return replace(notebook, hypotheses=new_hypotheses)


def add_exploration_step(
    notebook: DiagnosticNotebook,
    step: ExplorationStep,
) -> DiagnosticNotebook:
    """Add an exploration step to the notebook. Returns a new DiagnosticNotebook."""
    new_history = list(notebook.exploration_history) + [step]
    return replace(
        notebook,
        exploration_history=new_history,
        current_step=notebook.current_step + 1,
    )


def add_collected_data(
    notebook: DiagnosticNotebook,
    agent_id: str,
    data: dict,
) -> DiagnosticNotebook:
    """Add collected data from a Sub-Agent. Returns a new DiagnosticNotebook."""
    new_collected_data = {**notebook.collected_data, agent_id: data}
    return replace(notebook, collected_data=new_collected_data)


def set_confirmed_hypothesis(
    notebook: DiagnosticNotebook,
    hypothesis_id: str,
) -> DiagnosticNotebook:
    """Set the confirmed hypothesis. Returns a new DiagnosticNotebook."""
    return replace(notebook, confirmed_hypothesis=hypothesis_id)


def format_notebook_for_llm(notebook: DiagnosticNotebook) -> str:
    """Format the notebook into an LLM prompt string, using summaries for compressed phases."""
    lines: list[str] = []

    lines.append(f"# Diagnostic Notebook — Task: {notebook.task_id}")
    lines.append(f"Description: {notebook.task_description}")
    lines.append(f"Started: {notebook.start_time}")
    lines.append(f"Current Phase: {notebook.current_phase.value}")
    lines.append(f"Current Step: {notebook.current_step}")
    lines.append("")

    if notebook.phase_summaries:
        lines.append("## Phase Summaries")
        for summary in notebook.phase_summaries:
            lines.append(f"### Phase: {summary.phase} ({summary.started_at} → {summary.completed_at})")
            if summary.actions_taken:
                lines.append("Actions: " + ", ".join(summary.actions_taken))
            if summary.decisions_made:
                lines.append("Decisions: " + ", ".join(summary.decisions_made))
            if summary.hypotheses_affected:
                lines.append("Hypotheses affected: " + ", ".join(summary.hypotheses_affected))
        lines.append("")

    if notebook.hypotheses:
        lines.append("## Hypotheses")
        for h_id, h in notebook.hypotheses.items():
            lines.append(f"### [{h.status.value.upper()}] {h_id}: {h.description}")
            if h.evidence:
                lines.append("Evidence:")
                for e in h.evidence:
                    lines.append(f"  + {e}")
            if h.counter_evidence:
                lines.append("Counter-evidence:")
                for ce in h.counter_evidence:
                    lines.append(f"  - {ce}")
        lines.append("")

    if notebook.confirmed_hypothesis:
        lines.append(f"## Confirmed Root Cause: {notebook.confirmed_hypothesis}")
        lines.append("")

    if notebook.exploration_history:
        lines.append("## Exploration History")
        for step in notebook.exploration_history:
            lines.append(
                f"Step {step.step_number} [{step.phase.value}] {step.action} @ {step.timestamp}"
            )
            lines.append(f"  {step.content}")
        lines.append("")

    return "\n".join(lines)
