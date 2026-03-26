"""RCA-specific context formatters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentm.scenarios.rca.hypothesis_store import HypothesisStore
    from agentm.scenarios.rca.service_profile import ServiceProfileStore


def format_rca_context(
    *,
    profile_store: ServiceProfileStore | None = None,
    hypothesis_store: HypothesisStore | None = None,
) -> str:
    """Format RCA investigation state for the LLM context message.

    Reads from independent stores (hypothesis + service profiles).
    No state dict needed — stores are the source of truth.
    """
    sections: list[str] = []

    # Hypotheses (from store)
    if hypothesis_store is not None:
        hyp_section = hypothesis_store.format_for_llm()
        if hyp_section:
            sections.append(hyp_section)

    # Service profiles (from store)
    if profile_store is not None:
        profiles_section = profile_store.format_for_llm()
        if profiles_section:
            sections.append(profiles_section)

    if not sections:
        return "(Investigation starting — no data collected yet)"

    return "\n\n".join(sections)
