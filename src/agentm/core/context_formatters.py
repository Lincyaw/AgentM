"""Context formatters for the Node Orchestrator's system prompt working-memory section.

Each formatter receives the current graph state dict and returns a string that
is injected into the orchestrator's system prompt as the "working memory" block.

The FORMAT_CONTEXT_REGISTRY maps system_type -> callable (or None for types that
do not need a custom formatter).
"""

from __future__ import annotations

from typing import Callable

from agentm.core.compression import compress_completed_phase
from agentm.core.notebook import format_notebook_for_llm, should_compress_phase


# ---------------------------------------------------------------------------
# RCA / hypothesis-driven formatter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Memory-extraction formatter
# ---------------------------------------------------------------------------


def format_memory_extraction_context(state: dict) -> str:
    """Format MemoryExtractionState fields for LLM system prompt.

    Renders source_trajectories, extracted_patterns, knowledge_entries, and
    existing_knowledge as a readable text block.
    """
    lines: list[str] = []

    source_trajectories: list[str] = state.get("source_trajectories", [])
    if source_trajectories:
        lines.append("## Source Trajectories")
        for t in source_trajectories:
            lines.append(f"  - {t}")
        lines.append("")

    extracted_patterns: list[dict] = state.get("extracted_patterns", [])
    if extracted_patterns:
        lines.append(f"## Extracted Patterns ({len(extracted_patterns)} total)")
        for i, p in enumerate(extracted_patterns[:20], 1):
            ptype = p.get("pattern_type", "unknown")
            desc = p.get("description", "")
            lines.append(f"  {i}. [{ptype}] {desc}")
        if len(extracted_patterns) > 20:
            lines.append(f"  ... and {len(extracted_patterns) - 20} more")
        lines.append("")

    knowledge_entries: list = state.get("knowledge_entries", [])
    if knowledge_entries:
        lines.append(f"## Knowledge Entries Queued ({len(knowledge_entries)} total)")
        for entry in knowledge_entries[:10]:
            if hasattr(entry, "title"):
                title = entry.title
            elif isinstance(entry, dict):
                title = entry.get("title", str(entry))
            else:
                title = str(entry)
            lines.append(f"  - {title}")
        if len(knowledge_entries) > 10:
            lines.append(f"  ... and {len(knowledge_entries) - 10} more")
        lines.append("")

    existing_knowledge: list = state.get("existing_knowledge", [])
    if existing_knowledge:
        lines.append(f"## Existing Knowledge ({len(existing_knowledge)} entries)")
        for entry in existing_knowledge[:5]:
            if hasattr(entry, "title"):
                title = entry.title
            elif isinstance(entry, dict):
                title = entry.get("title", str(entry))
            else:
                title = str(entry)
            lines.append(f"  - {title}")
        if len(existing_knowledge) > 5:
            lines.append(f"  ... and {len(existing_knowledge) - 5} more")
        lines.append("")

    if not lines:
        return "(Memory extraction starting — no data yet)"

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORMAT_CONTEXT_REGISTRY: dict[str, Callable[[dict], str] | None] = {
    "hypothesis_driven": format_rca_context,
    "memory_extraction": format_memory_extraction_context,
    "sequential": None,
    "decision_tree": None,
}
