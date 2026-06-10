"""Single source of truth for the extractor child's preamble directive.

The directive prepended to every extractor child prompt
(workflow + finalize semantics + start-id rule + cross-firing reference
rule) MUST be byte-identical between the live seam
(:class:`llmharness.runtime.live.LiveChildRunner`) and the offline
seam (:class:`llmharness.runtime.offline.StandaloneChildRunner`).
Two different prompts would silently violate the design's invariant #1
(live === offline equivalence).

Keep this helper pure: payload in → text out, no I/O, no state.
"""

from __future__ import annotations

from typing import Any


def build_extractor_directive(payload: dict[str, Any]) -> str:
    """Return the directive preamble to prepend to the JSON payload.

    The text MUST match what every live extractor child has been seeing
    byte-for-byte; do not paraphrase. Steps are assembled as an ordered
    list and numbered by position, so inserting the optional
    tool-call-budget step cannot desync the trailing step numbers. The
    two payload-derived values are ``next_event_id`` (fixes the start id)
    and the ``recent_graph`` length (how many prior-firing entries the
    cross-firing edges may reference).
    """
    recent_n = len(payload.get("recent_graph") or [])
    next_id = payload.get("next_event_id")
    tool_call_budget = _positive_int_or_none(payload.get("tool_call_budget"))

    steps: list[str] = [
        "Build the graph incrementally with upsert_node / "
        "upsert_edge (and delete_node / delete_edge as needed). Every "
        "edit is validated immediately for witness + id rules. The "
        "validator may flag chain-link events (in=1, out=1) as a "
        "SOFT warning attached to a successful finalize — aim for "
        "compact graphs but do NOT fabricate refs just to satisfy "
        "the heuristic.\n",
        "Call finalize_extraction (no payload) when you are done. "
        "Finalize commits the witness-valid graph and ends the "
        "firing; any chain-link advisory comes back as part of the "
        "success result so the next firing can apply the hint.\n",
    ]
    if tool_call_budget is not None:
        edit_budget = max(tool_call_budget - 1, 0)
        steps.append(
            f"Tool-call budget: this extractor firing has at most "
            f"{tool_call_budget} total tool calls, including "
            "finalize_extraction. You MUST reserve the final tool call "
            "for finalize_extraction; do not spend the last call on a "
            "graph edit. Spend at most "
            f"{edit_budget} calls on graph edits, then call "
            "finalize_extraction immediately. If the graph is already "
            "coherent enough, prefer a smaller truthful graph and call "
            "finalize_extraction early instead of adding or revising more "
            "edges.\n"
        )
    steps.append(
        f"Start event ids at {next_id} and increment strictly — "
        "do NOT restart at 1 and do NOT reuse any id from recent_graph.\n"
    )
    steps.append(
        f"Cross-firing references: recent_graph has {recent_n} "
        "entries. To link this firing's events to prior firings, emit "
        "upsert_edge with src/dst spanning the boundary — the folded "
        "view already contains prior-firing nodes by id. Most act "
        "events in this firing answer a hyp/act from earlier firings; "
        "linking them is what turns a single firing into a connected "
        "investigation.\n\n"
    )

    numbered = "".join(f"({i}) {step}" for i, step in enumerate(steps, start=1))
    return "Below is the firing input. Workflow:\n" + numbered


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


__all__ = ["build_extractor_directive"]
