"""RCA-specific compression — phase compression for DiagnosticNotebook."""

from __future__ import annotations

from dataclasses import replace

from agentm.scenarios.rca.data import DiagnosticNotebook, PhaseSummary


def compress_completed_phase(
    notebook: DiagnosticNotebook,
    completed_phase: str,
) -> DiagnosticNotebook:
    """Compress a completed phase's detailed records into a PhaseSummary.

    Returns a new DiagnosticNotebook with phase_summaries updated and
    exploration_history pruned for the completed phase.
    """
    phase_steps = [
        step
        for step in notebook.exploration_history
        if step.phase.value == completed_phase
    ]
    remaining_steps = [
        step
        for step in notebook.exploration_history
        if step.phase.value != completed_phase
    ]

    started_at = phase_steps[0].timestamp if phase_steps else ""
    completed_at = phase_steps[-1].timestamp if phase_steps else ""
    actions_taken = [step.action for step in phase_steps]

    hypothesis_ids: list[str] = []
    for step in phase_steps:
        if (
            step.target_hypothesis_id
            and step.target_hypothesis_id not in hypothesis_ids
        ):
            hypothesis_ids.append(step.target_hypothesis_id)

    summary = PhaseSummary(
        phase=completed_phase,
        started_at=started_at,
        completed_at=completed_at,
        actions_taken=actions_taken,
        hypotheses_affected=hypothesis_ids,
    )

    new_phase_summaries = list(notebook.phase_summaries) + [summary]
    return replace(
        notebook,
        exploration_history=remaining_steps,
        phase_summaries=new_phase_summaries,
    )
