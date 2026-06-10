"""Three-section actionable error helper for the extractor tool surface.

Every error from the extractor tools renders the same three sections
(symptom, what-you-tried + current-graph, next-options) so the LLM
learns one template and can act on concrete tool-call suggestions.
"""

from __future__ import annotations


def format_witness_error(
    *,
    symptom: str,
    attempt: str | None,
    state_echo: str | None,
    options: list[str],
) -> str:
    """Render a three-section actionable error message."""
    if not options:
        raise ValueError(
            "format_witness_error: 'options' must be non-empty — every "
            "tool error must name at least one concrete next action"
        )
    attempt_line = attempt if attempt else "—"
    state_line = state_echo if state_echo else "(empty)"
    labelled: list[str] = []
    for idx, opt in enumerate(options):
        labelled.append(f"    ({chr(ord('a') + idx)}) {opt}")
    return (
        f"{symptom}\n"
        "\n"
        f"  what you tried:    {attempt_line}\n"
        f"  current graph:     {state_line}\n"
        "\n"
        "  next options:\n" + "\n".join(labelled)
    )


__all__ = ["format_witness_error"]
