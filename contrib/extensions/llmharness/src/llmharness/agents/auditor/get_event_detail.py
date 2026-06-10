"""``get_event_detail([ids])`` drill-down tool — Pydantic-backed.

Companion to :mod:`llmharness.audit.auditor.get_turn`. While ``get_turn``
drills down into the raw parent-session trajectory, ``get_event_detail``
drills down into the **audit graph** that the extractor produced (design
§4.g, §11).

Why this exists. When the auditor runs in *degraded* mode (more than
``audit_summary_threshold`` events — default 30, see :mod:`prompt`),
the prompt only carries each event's ``{id, kind, summary,
source_turns}`` and each edge's ``{src, dst, kind, reason}``. Witness
fields (``cited_entities``, ``cited_quote``) and full edge metadata are
stripped. The auditor pulls them on demand by calling
``get_event_detail([ids])`` with the event ids it wants the full records
for. The tool returns each event together with its outgoing- and
incoming-edge lists.

Contract for unknown ids. Per the v3 plan: unknown / out-of-range ids
are reflected by a top-level ``"missing"`` list (not a per-key
``null``). Known ids appear under their stringified id key. A request
with no known ids returns ``{"missing": [...]}``.

Out-of-shape input (e.g. negative id, non-int id, empty event_ids list)
returns a structured ``ToolResult`` with ``is_error=True`` rather than
raising — the auditor child loop must never crash from a malformed
drill-down call.

This module is **not** an atom — the merged ``atom.py`` calls
:func:`build_get_event_detail_tool` to mint a stateful
:class:`FunctionTool` over the configured events/edges at install time.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.lib import pydantic_to_tool_schema
from pydantic import BaseModel, ConfigDict, Field

from llmharness.runtime.decorator import harness_tool
from llmharness.schema import Edge, Event

GET_EVENT_DETAIL_TOOL_NAME = "get_event_detail"


class GetEventDetailArgs(BaseModel):
    # No class docstring — pydantic would emit it as top-level schema
    # ``description``; the hand-written V1 schema did not carry one.
    model_config = ConfigDict(extra="forbid", strict=True)

    event_ids: list[int] = Field(
        min_length=1,
        description=("Event ids to fetch full details for. Batched to amortize round trips."),
    )


# Stateless schema constant — exported for downstream training code that
# needs to register the tool surface without actual events/edges in hand.
GET_EVENT_DETAIL_PARAMETERS: dict[str, Any] = pydantic_to_tool_schema(GetEventDetailArgs)


def _coerce_events(raw: Any) -> tuple[Event, ...]:
    """Accept either a tuple of Event or a list of Event-shaped dicts."""
    if isinstance(raw, tuple) and all(isinstance(x, Event) for x in raw):
        return raw
    if isinstance(raw, list):
        out: list[Event] = []
        for item in raw:
            if isinstance(item, Event):
                out.append(item)
            elif isinstance(item, dict):
                try:
                    out.append(Event.from_dict(item))
                except (KeyError, ValueError, TypeError):
                    continue
        return tuple(out)
    return ()


def _coerce_edges(raw: Any) -> tuple[Edge, ...]:
    if isinstance(raw, tuple) and all(isinstance(x, Edge) for x in raw):
        return raw
    if isinstance(raw, list):
        out: list[Edge] = []
        for item in raw:
            if isinstance(item, Edge):
                out.append(item)
            elif isinstance(item, dict):
                try:
                    out.append(Edge.from_dict(item))
                except (KeyError, ValueError, TypeError):
                    continue
        return tuple(out)
    return ()


def _err(text: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=True,
    )


def build_get_event_detail_tool(
    events: tuple[Event, ...] | list[Event] | list[dict[str, Any]],
    edges: tuple[Edge, ...] | list[Edge] | list[dict[str, Any]],
) -> FunctionTool:
    """Mint a :class:`FunctionTool` closing over ``events`` / ``edges``."""
    events_t = _coerce_events(events)
    edges_t = _coerce_edges(edges)

    events_by_id: dict[int, Event] = {ev.id: ev for ev in events_t}
    outgoing: dict[int, list[Edge]] = {}
    incoming: dict[int, list[Edge]] = {}
    for ed in edges_t:
        outgoing.setdefault(ed.src, []).append(ed)
        incoming.setdefault(ed.dst, []).append(ed)

    @harness_tool(GET_EVENT_DETAIL_TOOL_NAME)
    async def _get_event_detail(args: GetEventDetailArgs, _ctx: Any) -> ToolResult:
        """Fetch the full Event.to_dict() plus outgoing and incoming Edge.to_dict() lists for every event id in event_ids. Use in degraded-prompt mode (more events than audit_summary_threshold) to recover witness fields. Unknown ids appear in a top-level "missing" list. Out-of-shape input returns a structured tool-result error rather than raising."""
        # bool is a subclass of int in Python; strict mode in pydantic rejects
        # it, but defend the boundary anyway.
        for x in args.event_ids:
            if isinstance(x, bool):
                return _err(
                    "get_event_detail rejected: every event_ids entry must "
                    "be an integer, got 'bool'"
                )
            if x < 0:
                return _err(f"get_event_detail rejected: negative id {x} is invalid")

        result: dict[str, Any] = {}
        missing: list[int] = []
        for eid in args.event_ids:
            ev = events_by_id.get(eid)
            if ev is None:
                missing.append(eid)
                continue
            result[str(eid)] = {
                "event": ev.to_dict(),
                "outgoing_edges": [e.to_dict() for e in outgoing.get(eid, [])],
                "incoming_edges": [e.to_dict() for e in incoming.get(eid, [])],
            }
        if missing:
            result["missing"] = missing
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False),
                )
            ],
            is_error=False,
        )

    return _get_event_detail


__all__ = [
    "GET_EVENT_DETAIL_PARAMETERS",
    "GET_EVENT_DETAIL_TOOL_NAME",
    "GetEventDetailArgs",
    "build_get_event_detail_tool",
]
