"""Three-section actionable error helper for the extractor tool surface.

Why this exists. An OpenAI rca:harness.sync run on 2026-05-21 showed the
extractor's witness validator rejecting the same passthrough payload 6
times in a row. Every rejection used the same generic phrasing ("merge
or promote to branch") with no concrete tool-call suggestion, so the
LLM treated each retry as a new puzzle rather than as a directive. By
turn 7 it accepted a degraded graph (2 events dropped) just to escape
the loop — terrible SFT signal.

The fix is structural: every error from the extractor tools must render
the *same* three sections so the model learns one template:

    <symptom — one line>

      what you tried:    <args echo>
      current graph:     <state summary>

      next options:
        (a) <concrete tool call with concrete ids>
        (b) ...

The helper deliberately stays plain-string — JSON would force the LLM
into structured parsing and providers already wrap the text. Concrete
tool names + ids in options are the part that flipped retry behaviour
in the calibration runs; abstract advice does not.

Not an atom — sibling to ``_tool_decorator.py``. Imported from each
tool handler module under ``audit/extractor/``.
"""

from __future__ import annotations


def format_witness_error(
    *,
    symptom: str,
    attempt: str | None,
    state_echo: str | None,
    options: list[str],
) -> str:
    """Render a three-section actionable error message.

    Sections, in order:

    * **symptom** — single line stating what failed. No trailing period
      style preference; pass the raw sentence.
    * **what you tried / current graph** — two short labelled lines so
      the model sees the rejection in the context of its own action and
      of the accepted state. ``None`` for either renders as the literal
      ``—`` / ``(empty)`` placeholder so absence is visible.
    * **next options** — at least one concrete suggestion, formatted as
      ``(a)`` / ``(b)`` / ... with the tool name and concrete values.

    Empty ``options`` raises :class:`ValueError` at format time —
    silently returning an unactionable error is the failure mode this
    helper exists to prevent.
    """
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
