"""RCA-specific context formatters."""

from __future__ import annotations

from agentm.scenarios.rca.compression import compress_completed_phase
from agentm.scenarios.rca.notebook import format_notebook_for_llm, should_compress_phase


def format_rca_context(state: dict) -> str:
    """Format HypothesisDrivenState notebook for LLM system prompt.

    Applies phase compression to completed phases before formatting.
    Returns a human-readable notebook text block.
    """
    notebook = state.get("notebook")
    if notebook is None:
        return "(Investigation starting — no data collected yet)"

    notebook_for_llm = notebook
    for phase in ("exploration", "generation", "verification"):
        if should_compress_phase(notebook_for_llm, phase):
            notebook_for_llm = compress_completed_phase(notebook_for_llm, phase)

    return format_notebook_for_llm(notebook_for_llm)
