"""Build the v3.1 extractor's single ``submit_events`` tool.

V3 (issue #134) had three tools — ``register_event``, ``add_edge``,
``submit_extraction`` — with multi-turn dance per firing (LLM called
each tool separately, retrying ``add_edge`` per witness failure). That
took 8-12 turns per extractor child session and dominated audit
overhead.

V3.1 collapses them into one tool: the LLM produces a single JSON
payload listing events with embedded ``refs[]``. Validation runs in
:meth:`ExtractionState.commit`. The child loop terminates immediately
with ``ToolTerminate``. Hard-reject errors (event-shape) are returned
as a tool error so the LLM may retry within the caller's attempt
budget; partial-success drops are recorded into ``dropped_edges`` and
the submission is accepted.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate

from .._enum_schema import EDGE_KIND_VALUES, EVENT_KIND_VALUES
from .state import ExtractionState

SUBMIT_EVENTS_TOOL_NAME = "submit_events"
SUBMIT_EVENTS_REASON = "llmharness:submit_events"

EXTRACTOR_TOOL_NAMES: tuple[str, ...] = (SUBMIT_EVENTS_TOOL_NAME,)


_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "to": {
            "type": "integer",
            "description": (
                "id of an EARLIER event in this submission (must be < the "
                "containing event's id). The referenced event is the source "
                "of this relation; the containing event is the destination."
            ),
        },
        "kind": {
            "type": "string",
            "enum": list(EDGE_KIND_VALUES),
            "description": (
                "'data' = content/data flow (requires cited_entities); "
                "'ref' = referential mention (requires cited_quote)."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining the connection.",
        },
        "cited_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Required non-empty when kind='data'. Each entity must "
                "appear (case+ws normalized substring) in BOTH the source "
                "event's source_turns text and the citing event's "
                "source_turns text."
            ),
        },
        "cited_quote": {
            "type": "string",
            "description": (
                "Required non-empty when kind='ref'. Must appear "
                "(case+ws normalized substring) in BOTH the source event's "
                "and the citing event's source_turns text."
            ),
        },
    },
    "required": ["to", "kind", "reason"],
    "additionalProperties": False,
}


_EXTERNAL_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "to_recent_graph_index": {
            "type": "integer",
            "description": (
                "1-based index into the recent_graph slice the harness "
                "presented this firing. Identifies which prior event is "
                "the source of this cross-firing relation."
            ),
        },
        "kind": {
            "type": "string",
            "enum": list(EDGE_KIND_VALUES),
            "description": (
                "Same semantics as refs[].kind. 'data' = data flow "
                "(requires cited_entities); 'ref' = referential mention "
                "(requires cited_quote)."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining the connection.",
        },
        "cited_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Required non-empty when kind='data'. Each entity must "
                "appear (case+ws normalized substring) in BOTH the prior "
                "event's source_turns text and the citing event's "
                "source_turns text."
            ),
        },
        "cited_quote": {
            "type": "string",
            "description": (
                "Required non-empty when kind='ref'. Must appear "
                "(case+ws normalized substring) in BOTH the prior event's "
                "and the citing event's source_turns text."
            ),
        },
    },
    "required": ["to_recent_graph_index", "kind", "reason"],
    "additionalProperties": False,
}


_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
            "description": (
                "Local-firing id starting at 1. Events MUST be submitted in "
                "id order: events[0].id=1, events[1].id=2, ..., with no "
                "gaps. ids in this firing are independent of any other "
                "firing's ids — recent_graph ids are read-only context."
            ),
        },
        "kind": {
            "type": "string",
            "enum": list(EVENT_KIND_VALUES),
            "description": (
                "Closed-set event kind classified by ACTION SIGNATURE, "
                "not by what the agent says it is doing."
            ),
        },
        "summary": {
            "type": "string",
            "description": "≤ 30 words describing what happened.",
        },
        "source_turns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Trajectory indices this event was extracted from. Must "
                "be non-empty; witnesses on this event's refs are checked "
                "against the concatenated text of these turns."
            ),
        },
        "refs": {
            "type": "array",
            "items": _REF_SCHEMA,
            "description": (
                "References this event makes to EARLIER events (smaller "
                "id within THIS firing). Always required as a field; "
                "may be empty when the event only cites prior firings "
                "via external_refs. Events with id>=2 must cite at "
                "least one earlier event (in-firing OR external) — the "
                "validator rejects events that have neither."
            ),
        },
        "external_refs": {
            "type": "array",
            "items": _EXTERNAL_REF_SCHEMA,
            "description": (
                "Cross-firing references this event makes back into the "
                "recent_graph slice the harness presented. Each entry "
                "names a prior event by its 1-based index in recent_graph "
                "and carries the same witness shape as refs[]. Use these "
                "when an event in this firing is causally connected to a "
                "prior firing's event (e.g. a tool result evid here "
                "answers a hypothesis emitted two firings ago). The "
                "offline aggregator resolves these to edges in the "
                "cumulative global id space. Optional; default empty."
            ),
        },
    },
    "required": ["id", "kind", "summary", "source_turns", "refs"],
    "additionalProperties": False,
}


_SUBMIT_EVENTS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": _EVENT_SCHEMA,
            "description": (
                "All events extracted from the new turn window, in id "
                "order (1..N). May be empty if the window has no "
                "semantically meaningful moves."
            ),
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


def _ok(payload: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=payload)])


def _err(payload: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=payload)], is_error=True)


def build_extractor_tools(
    state: ExtractionState, *, witness_retry_budget: int = 0
) -> list[FunctionTool]:
    """Build the v3.1 single-tool extractor surface, closed over ``state``.

    ``witness_retry_budget`` controls how many times a submission with
    non-empty ``dropped_edges`` is bounced back as a tool error so the
    LLM can fix the offending ``cited_entities`` / ``cited_quote`` and
    re-submit. Default ``0`` preserves the V3.1 single-shot contract
    (drops are accepted silently). Set to ``1`` for one bounded retry
    — caps total LLM calls per firing at 2 instead of V3's 8-12 while
    recovering most semantically-valid but literally-mis-cited refs.
    Hard-reject errors (event-shape) are always retryable via the
    kernel's attempt budget; this only governs soft (witness) drops.
    """

    attempts_used = 0

    async def _submit_events(args: dict[str, Any]) -> ToolResult | ToolTerminate:
        nonlocal attempts_used
        events_payload = args.get("events")
        if not isinstance(events_payload, list):
            return _err("submit_events: 'events' must be an array (may be empty)")
        err = state.commit(events_payload)
        if err is not None:
            # Hard reject — return as tool error so the LLM may retry
            # within the caller's attempt budget.
            return _err(err)

        # Soft drops: bounce back once if budget allows so the LLM can
        # fix entity selection. Reset the committed flag + frozen
        # results so the second commit overwrites cleanly.
        if state.dropped_edges and attempts_used < witness_retry_budget:
            dropped_snapshot = list(state.dropped_edges)
            state.committed = False
            state.events = ()
            state.edges = ()
            state.dropped_edges = ()
            attempts_used += 1
            feedback = _format_witness_feedback(dropped_snapshot)
            return _err(feedback)

        # Success (full or partial). Echo a concise digest so the LLM
        # has a deterministic stop signal.
        digest = (
            f'{{"ok": true, "events": {len(state.events)}, '
            f'"edges": {len(state.edges)}, '
            f'"dropped": {len(state.dropped_edges)}}}'
        )
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text=digest)]),
            reason=SUBMIT_EVENTS_REASON,
        )

    return [
        FunctionTool(
            name=SUBMIT_EVENTS_TOOL_NAME,
            description=(
                "Submit the entire event graph for this firing in ONE call. "
                "Each event carries TWO load-bearing ref lists: ``refs[]`` "
                "for connections to earlier events in THIS firing, and "
                "``external_refs[]`` for connections back into "
                "``recent_graph`` (prior firings). Both are witness-validated "
                "against literal turn texts; refs that fail witness are "
                "dropped while the events stay. Skipping ``external_refs`` "
                "leaves the cumulative graph as disconnected per-firing "
                "islands — emit them whenever a literal token appears in "
                "both a recent_graph entry's source_turn_texts and this "
                "event's source_turns text. Event-shape errors hard-reject "
                "the whole submission and you may retry. Call this exactly "
                "ONCE as your final action."
            ),
            parameters=_SUBMIT_EVENTS_PARAMETERS,
            fn=_submit_events,
        ),
    ]


def _format_witness_feedback(dropped: list[dict[str, Any]]) -> str:
    """Render a structured retry directive listing every dropped ref.

    Surfaces each ``last_error`` so the LLM knows which entity / quote
    failed and on which side, lets it locate the literal token in the
    turn text, and re-submit the entire ``events`` payload with the
    correction. Keep the message terse so it fits the next prompt
    cleanly.
    """
    lines = [
        "submit_events: witness failed on "
        f"{len(dropped)} ref(s). Re-submit the FULL events payload with "
        "corrected cited_entities / cited_quote. Each failed ref below "
        "lists src -> dst and the validator's diagnostic; replace the "
        "cited token with the exact literal substring that appears in "
        "BOTH source_turns texts after case+whitespace normalization, "
        "or drop the ref if no shared literal token exists.",
        "",
    ]
    for d in dropped:
        src = d.get("src")
        dst = d.get("dst")
        kind = d.get("kind")
        err = d.get("last_error") or ""
        lines.append(f"- {src} -> {dst} ({kind}): {err}")
    return "\n".join(lines)


__all__ = [
    "EXTRACTOR_TOOL_NAMES",
    "SUBMIT_EVENTS_REASON",
    "SUBMIT_EVENTS_TOOL_NAME",
    "build_extractor_tools",
]
