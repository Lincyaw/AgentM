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

SUBMIT_PLAN_TOOL_NAME = "submit_plan"
GRAPH_EDIT_TOOL_NAME = "graph_edit"
SUBMIT_EVENTS_BATCH_TOOL_NAME = "submit_events_batch"
RESET_EXTRACTION_TOOL_NAME = "reset_extraction"
SUBMIT_EVENTS_REASON = "llmharness:submit_events_batch_done"

# Legacy aliases retained for callers that haven't migrated to the
# v18 two-tool flow yet — the adapter still inspects these names when
# computing whether the extractor ran at all.
SUBMIT_EVENTS_TOOL_NAME = SUBMIT_EVENTS_BATCH_TOOL_NAME

EXTRACTOR_TOOL_NAMES: tuple[str, ...] = (
    SUBMIT_PLAN_TOOL_NAME,
    GRAPH_EDIT_TOOL_NAME,
    SUBMIT_EVENTS_BATCH_TOOL_NAME,
    RESET_EXTRACTION_TOOL_NAME,
)


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
        "to_recent_event_id": {
            "type": "integer",
            "description": (
                "Global event id of the prior event you are referencing — "
                "copy the value of recent_graph[i].id verbatim. Identifies "
                "which prior event is the source of this cross-firing relation. "
                "Do NOT use the array position; use the .id field."
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
    "required": ["to_recent_event_id", "kind", "reason"],
    "additionalProperties": False,
}


_BLOCK_PLAN_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Contiguous trajectory indices that form this block. A "
                "linear block (one target, no branching) typically spans "
                "several turns; a branch-point block (hyp/dec/concl) is "
                "usually one turn. Every turn in the new-turn window "
                "appears in at least one block. Blocks are normally "
                "disjoint, with one specific exception: a choice-point "
                "branch block (a hyp formed by the agent's targeting "
                "choice at a tool_call turn) may share its single turn "
                "with the first turn of the linear block that executes "
                "the choice — see the passthrough worked example in the "
                "prompt's 'The block plan' section."
            ),
        },
        "kind": {
            "type": "string",
            "enum": ["linear", "branch"],
            "description": (
                "'linear': the agent is probing one target without "
                "branching — emit ONE act + ONE evid covering all of "
                "the block's turns. 'branch': a reasoning move "
                "(hyp/dec/concl) — emit ONE atomic event."
            ),
        },
        "note": {
            "type": "string",
            "description": (
                "One short sentence naming the target/intent of the "
                "block. Linear: which service / file / data class is "
                "being probed. Branch: which kind of move (hyp/dec/"
                "concl) and what it's about."
            ),
        },
    },
    "required": ["turns", "kind", "note"],
    "additionalProperties": False,
}


_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
            "description": (
                "GLOBAL event id. Start at the ``next_event_id`` value in "
                "the payload and increment strictly (events[0].id = "
                "next_event_id, events[1].id > events[0].id, ...). The id "
                "namespace is shared with prior firings; do NOT reuse any "
                "id present in recent_graph and do NOT restart at 1."
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
            "description": (
                "Natural-language paragraph describing this event, with "
                "LENGTH PROPORTIONAL TO source_turns COUNT. A "
                "single-turn branch event (task / hyp / dec / concl) is "
                "one focused sentence with the concrete claim. A linear "
                "act or evid that COALESCES N consecutive turns must be "
                "a paragraph that walks through what happened across "
                "those N turns: roughly one short sentence per covered "
                "turn (so a 10-turn block ≈ 10 sentences ≈ 100-130 "
                "words; a 20-turn block ≈ 20 sentences ≈ 200-250 "
                "words). Name every distinct tool_call's concrete "
                "parameters verbatim (services, time windows, query "
                "filters, file paths, error codes, span/log/metric "
                "names) and quote the key numbers each result returned "
                "(row counts, latencies, error counts, ratios). A "
                "20-turn act whose summary is 30 words has thrown away "
                "the very signal the downstream auditor needs — it is a "
                "compression failure, not a tight summary."
            ),
        },
        "source_turns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Trajectory indices this event was extracted from. Must "
                "be non-empty AND contiguous (a single range "
                "[first, first+1, ..., last] with no gaps). For an "
                "act+evid pair coalesced from one linear block of "
                "turns N..M, BOTH events list the full range "
                "[N, N+1, ..., M]. Do NOT split tool_call turns to "
                "act and tool_result turns to evid — the kinds "
                "represent two narrative facets of the same time "
                "segment, not a partition of turn types. Witnesses on "
                "this event's refs are checked against the "
                "concatenated text of these turns."
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
                "Cross-firing references this event makes back into "
                "``recent_graph`` (the full prior graph the harness "
                "presented). Each entry names a prior event by its "
                "global id via ``to_recent_event_id`` (copy "
                "``recent_graph[i].id`` verbatim — NOT the array index) "
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


_SUBMIT_PLAN_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "block_plan": {
            "type": "array",
            "items": _BLOCK_PLAN_ENTRY_SCHEMA,
            "description": (
                "Partition the new-turn window into contiguous basic "
                "blocks (see Principle 3 of the extractor prompt). Each "
                "turn in the window appears in at least one block. The "
                "plan is informational — the auditable invariant (every "
                "internal event must be a true branch point, not a "
                "passthrough) is enforced on the emitted events when "
                "you call submit_events_batch with done=true. Plan-"
                "structure enforcement (e.g. no two adjacent linear "
                "blocks) is intentionally OFF in v18; concentrate the "
                "discipline on the events you actually emit."
            ),
        },
    },
    "required": ["block_plan"],
    "additionalProperties": False,
}


_SUBMIT_EVENTS_BATCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": _EVENT_SCHEMA,
            "description": (
                "A batch of events to append to the firing's pending "
                "graph. Each batch is validated standalone (shape, id "
                "monotonicity, ref witnesses) and either accepted in "
                "full or rejected in full. Previously-accepted batches "
                "stay accepted across rejections — the LLM only needs "
                "to retry the batch that failed, not the whole graph. "
                "Ids must continue the global id sequence: the first "
                "event of the first batch starts at next_event_id, and "
                "subsequent events / batches increment strictly. Refs "
                "may point at events from earlier batches or earlier "
                "in this same batch."
            ),
        },
        "done": {
            "type": "boolean",
            "description": (
                "Set true ONLY on the final batch — that's when the "
                "cross-graph degree check runs (every internal event "
                "must be a true branch point, no passthroughs) and the "
                "firing terminates. If degree check fails, the batch "
                "is still accepted but the firing is NOT terminated; "
                "you may submit additional batches that add refs to "
                "promote passthrough events into branches, or call "
                "reset_extraction to start over."
            ),
        },
    },
    "required": ["events", "done"],
    "additionalProperties": False,
}


_RESET_EXTRACTION_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


_EDGE_SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "src": {"type": "integer"},
        "dst": {"type": "integer"},
        "kind": {"type": "string", "enum": list(EDGE_KIND_VALUES)},
    },
    "additionalProperties": False,
}


_GRAPH_EDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "src": {"type": "integer"},
        "dst": {"type": "integer"},
        "kind": {"type": "string", "enum": list(EDGE_KIND_VALUES)},
        "reason": {"type": "string"},
        "cited_entities": {"type": "array", "items": {"type": "string"}},
        "cited_quote": {"type": "string"},
    },
    "required": ["src", "dst", "kind", "reason"],
    "additionalProperties": False,
}


_GRAPH_EDIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": [
                "add_node",
                "update_node",
                "delete_node",
                "add_edge",
                "update_edge",
                "delete_edge",
            ],
            "description": "One graph mutation to apply to the pending extractor graph.",
        },
        "node": {
            "type": "object",
            "description": (
                "Full node for add_node, or patch for update_node. Node fields "
                "match submit_events_batch events: id, kind, summary, source_turns."
            ),
        },
        "node_id": {"type": "integer", "description": "Target node id for update/delete."},
        "edge": {
            "oneOf": [_GRAPH_EDGE_SCHEMA],
            "description": "Full edge for add_edge, or patch for update_edge.",
        },
        "edge_selector": {
            "oneOf": [_EDGE_SELECTOR_SCHEMA],
            "description": "Select an existing pending edge for update/delete.",
        },
    },
    "required": ["op"],
    "additionalProperties": False,
}


def _ok(payload: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=payload)])


def _err(payload: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=payload)], is_error=True)


def build_extractor_tools(
    state: ExtractionState, *, witness_retry_budget: int = 0
) -> list[FunctionTool]:
    """Build the v18 three-tool extractor surface, closed over ``state``.

    Tools returned:

    * ``submit_plan`` — called once, captures the block plan. Informational
      only; the v17 "no adjacent linear blocks" check was superseded by
      the v18 cross-graph degree check (see below). The plan still
      provides offline diagnostic value and gives the LLM a place to
      front-load CoT before emitting events.
    * ``submit_events_batch`` — called one OR more times. Each batch is
      validated standalone and either accepted in full (appended to
      pending) or rejected in full (LLM retries the batch). When called
      with ``done=true``, runs the cross-graph degree check (every
      internal event must be a true branch point, no passthroughs) and
      terminates the firing on success.
    * ``reset_extraction`` — drops pending state so the LLM can start
      over after an unrecoverable finalize rejection.

    ``witness_retry_budget`` controls how many times a batch with non-
    empty witness drops is bounced back so the LLM can fix the offending
    ``cited_entities`` / ``cited_quote``. Default ``0`` accepts drops
    silently. Hard-reject errors (event-shape) are always retryable via
    the kernel's attempt budget; this only governs soft witness drops.
    """

    attempts_used = 0

    async def _submit_plan(args: dict[str, Any]) -> ToolResult:
        plan_payload = args.get("block_plan")
        if plan_payload is None:
            return _err("submit_plan: 'block_plan' is required")
        if not isinstance(plan_payload, list):
            return _err("submit_plan: 'block_plan' must be an array")
        err = state.commit_plan(plan_payload)
        if err is not None:
            return _err(err)
        return _ok(f'{{"ok": true, "blocks": {len(state.block_plan)}}}')

    async def _submit_events_batch(
        args: dict[str, Any],
    ) -> ToolResult | ToolTerminate:
        nonlocal attempts_used
        events_payload = args.get("events")
        if not isinstance(events_payload, list):
            return _err(
                "submit_events_batch: 'events' must be an array (may be empty)"
            )
        done = bool(args.get("done", False))

        prev_dropped = len(state._dropped_pending)
        err = state.commit_batch(events_payload)
        if err is not None:
            return _err(err)

        # Soft drops: bounce back once per firing if budget allows so
        # the LLM can fix witness selection on the most-recent batch.
        new_drops = state._dropped_pending[prev_dropped:]
        if new_drops and attempts_used < witness_retry_budget:
            # Roll back the just-accepted batch so the retry can
            # overwrite it cleanly. This preserves the partial-batch
            # accept invariant for OTHER batches (only this one rolls).
            accepted_count = len(events_payload)
            state._events_pending = state._events_pending[:-accepted_count]
            # Edges + external_refs from this batch are interleaved with
            # earlier batches' state; reconstruct by filtering.
            accepted_ids = {
                ev["id"] for ev in events_payload if isinstance(ev, dict)
            }
            state._edges_pending = [
                e for e in state._edges_pending if e.dst not in accepted_ids
            ]
            for eid in list(state._external_refs_pending):
                if eid in accepted_ids:
                    state._external_refs_pending.pop(eid, None)
            state._dropped_pending = state._dropped_pending[:prev_dropped]
            attempts_used += 1
            return _err(_format_witness_feedback(list(new_drops)))

        if not done:
            digest = (
                f'{{"ok": true, "batch_events": {len(events_payload)}, '
                f'"pending_events": {len(state._events_pending)}, '
                f'"pending_edges": {len(state._edges_pending)}, '
                f'"pending_dropped": {len(state._dropped_pending)}, '
                '"done": false}'
            )
            return _ok(digest)

        # done=true: run finalize. On degree-check failure the batch
        # stays accepted but the firing is NOT terminated — the LLM
        # can submit more batches that promote passthrough events.
        finalize_err = state.finalize()
        if finalize_err is not None:
            return _err(finalize_err)

        digest = (
            f'{{"ok": true, "events": {len(state.events)}, '
            f'"edges": {len(state.edges)}, '
            f'"dropped": {len(state.dropped_edges)}, '
            '"done": true}'
        )
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text=digest)]),
            reason=SUBMIT_EVENTS_REASON,
        )

    async def _graph_edit(args: dict[str, Any]) -> ToolResult:
        op = str(args.get("op") or "")
        result = state.apply_graph_edit(op, args)
        if isinstance(result, str):
            return _err(result)
        import json

        return _ok(json.dumps(result, ensure_ascii=False))

    async def _reset_extraction(args: dict[str, Any]) -> ToolResult:
        del args
        if state.committed:
            return _err(
                "reset_extraction: firing already finalized; reset is "
                "not possible after a successful submit_events_batch with done=true"
            )
        state.reset_pending()
        return _ok('{"ok": true, "reset": true}')

    return [
        FunctionTool(
            name=SUBMIT_PLAN_TOOL_NAME,
            description=(
                "Commit the firing's basic-block plan. Call this ONCE "
                "before any submit_events_batch. The plan partitions the "
                "new-turn window into contiguous blocks (linear vs "
                "branch); it's informational, not enforced as a "
                "structural constraint. The auditable invariant lives "
                "on the emitted events: every internal event must be a "
                "true branch point (in-degree > 1 OR out-degree > 1) — "
                "checked when you call submit_events_batch with "
                "done=true."
            ),
            parameters=_SUBMIT_PLAN_PARAMETERS,
            fn=_submit_plan,
        ),
        FunctionTool(
            name=GRAPH_EDIT_TOOL_NAME,
            description=(
                "Apply one direct edit to the pending extractor graph. "
                "Use this when you need to revise the graph incrementally: "
                "add/update/delete event nodes or witness-bearing edges. "
                "After graph_edit operations, call submit_events_batch with "
                "events=[] and done=true to finalize the edited pending graph."
            ),
            parameters=_GRAPH_EDIT_PARAMETERS,
            fn=_graph_edit,
        ),
        FunctionTool(
            name=SUBMIT_EVENTS_BATCH_TOOL_NAME,
            description=(
                "Append a batch of events to the firing's pending graph. "
                "May be called multiple times — accepted batches "
                "accumulate. Each batch is validated standalone: shape, "
                "monotonic ids, ref witnesses. Hard-reject errors fail "
                "ONLY this batch (previously accepted batches stay "
                "accepted); the LLM may retry the failed batch with "
                "corrections. Witness failures drop the offending refs "
                "and accept the rest of the batch. Set done=true on the "
                "final batch to trigger the cross-graph degree check "
                "(passthrough rejection) and terminate the firing. If "
                "the degree check rejects, the firing stays alive and "
                "you may submit additional batches that promote "
                "passthrough events into branches (e.g. by ref-ing them "
                "from a new event)."
            ),
            parameters=_SUBMIT_EVENTS_BATCH_PARAMETERS,
            fn=_submit_events_batch,
        ),
        FunctionTool(
            name=RESET_EXTRACTION_TOOL_NAME,
            description=(
                "Drop all pending events / edges so you can re-emit the "
                "firing's graph from scratch. Use this only when the "
                "accumulated graph is unrecoverable (e.g. you can't see "
                "a way to fix passthrough events without merging "
                "neighbours, which append-only can't do). The plan is "
                "preserved — only events / edges / drops are cleared."
            ),
            parameters=_RESET_EXTRACTION_PARAMETERS,
            fn=_reset_extraction,
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
    "GRAPH_EDIT_TOOL_NAME",
    "RESET_EXTRACTION_TOOL_NAME",
    "SUBMIT_EVENTS_BATCH_TOOL_NAME",
    "SUBMIT_EVENTS_REASON",
    "SUBMIT_EVENTS_TOOL_NAME",
    "SUBMIT_PLAN_TOOL_NAME",
    "build_extractor_tools",
]
