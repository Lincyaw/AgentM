"""Single source of truth for the extractor child's preamble directive.

The four-step directive prepended to every extractor child prompt
(workflow + finalize semantics + start-id rule + cross-firing reference
rule) MUST be byte-identical between the live seam
(:class:`llmharness.audit._live_seams.LiveChildRunner`) and the offline
seam (:class:`llmharness.audit._offline_seams.StandaloneChildRunner`).
Two different prompts would silently violate the design's invariant #1
(live === offline equivalence).

Keep this helper pure: payload in → text out, no I/O, no state.
"""

from __future__ import annotations

from typing import Any


def build_extractor_directive(payload: dict[str, Any]) -> str:
    """Return the four-step directive preamble to prepend to the JSON payload.

    The text below MUST match what every live extractor child has been
    seeing byte-for-byte; do not paraphrase. The two payload-derived
    values are:

    * ``next_event_id`` — used in step (3) to fix the start id.
    * ``recent_graph`` length — used in step (4) to spell out how many
      prior-firing entries the cross-firing edges may reference.
    """
    recent_n = len(payload.get("recent_graph") or [])
    next_id = payload.get("next_event_id")
    return (
        "Below is the firing input. Workflow:\n"
        "(1) Build the graph incrementally with upsert_node / "
        "upsert_edge (and delete_node / delete_edge as needed). Every "
        "edit is validated immediately; errors come back as a "
        "three-section actionable message naming concrete next "
        "options. Internal events must be true branch points "
        "(in-degree>=2 or out-degree>=2); passthrough (in=1, out=1) "
        "events are rejected at finalize.\n"
        "(2) Call finalize_extraction (no payload) when you are done. "
        "On a clean degree check the firing terminates; on rejection "
        "the firing stays alive so you can promote passthrough nodes "
        "with further upserts and re-call.\n"
        f"(3) Start event ids at {next_id} and increment strictly — "
        "do NOT restart at 1 and do NOT reuse any id from recent_graph.\n"
        f"(4) Cross-firing references: recent_graph has {recent_n} "
        "entries. To link this firing's events to prior firings, emit "
        "upsert_edge with src/dst spanning the boundary — the folded "
        "view already contains prior-firing nodes by id. Most evid "
        "events in this firing answer a hyp/act from earlier firings; "
        "linking them is what turns a single firing into a connected "
        "investigation.\n\n"
    )


__all__ = ["build_extractor_directive"]
