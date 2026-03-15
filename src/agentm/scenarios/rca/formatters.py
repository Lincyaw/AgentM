"""RCA-specific context formatters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentm.scenarios.rca.compression import compress_completed_phase
from agentm.scenarios.rca.notebook import format_notebook_for_llm, should_compress_phase

if TYPE_CHECKING:
    from agentm.scenarios.rca.service_profile import ServiceProfileStore


def format_rca_context(
    state: dict,
    *,
    profile_store: ServiceProfileStore | None = None,
) -> str:
    """Format HypothesisDrivenState notebook for LLM system prompt.

    Applies phase compression to completed phases before formatting.
    Returns a human-readable notebook text block.
    """
    notebook = state.get("notebook")
    if notebook is None:
        base = "(Investigation starting — no data collected yet)"
    else:
        notebook_for_llm = notebook
        for phase in ("exploration", "generation", "verification"):
            if should_compress_phase(notebook_for_llm, phase):
                notebook_for_llm = compress_completed_phase(notebook_for_llm, phase)
        base = format_notebook_for_llm(notebook_for_llm)

    if profile_store is not None:
        profiles_section = profile_store.format_for_llm()
        if profiles_section:
            base = base + "\n\n" + profiles_section

    return base
