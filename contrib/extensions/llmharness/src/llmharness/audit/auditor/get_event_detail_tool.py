"""§11 single-file extension: register the ``get_event_detail([ids])`` tool.

Companion to :mod:`llmharness.audit.auditor.get_turn_tool`. While
``get_turn`` drills down into the raw parent-session trajectory,
``get_event_detail`` drills down into the **audit graph** that the
extractor produced (design §4.g, §11).

Why this exists. When the auditor runs in *degraded* mode (more than
``audit_summary_threshold`` events — default 30, see :mod:`prompt`),
the prompt only carries each event's ``{id, kind, summary,
source_turns}`` and each edge's ``{src, dst, kind, reason}``.
Witness fields (``cited_entities``, ``cited_quote``) and full edge
metadata are stripped. The auditor pulls them on demand by calling
``get_event_detail([ids])`` with the event ids it wants the full
records for. The tool returns each event together with its
outgoing- and incoming-edge lists.

Contract for unknown ids. Per the v3 plan: unknown / out-of-range
ids are reflected by a top-level ``"missing"`` list (not a per-key
``null``). Known ids appear under their stringified id key. A request
with no known ids returns ``{"missing": [...]}``.

Out-of-shape input (e.g. negative id, non-int id, empty event_ids
list) returns a structured ``ToolResult`` with ``is_error=True``
rather than raising — the auditor child loop must never crash from
a malformed drill-down call.

Bridging matches the commit-3 pattern: the adapter passes ``events``
and ``edges`` through ``compose_auditor_extensions(...)`` which
forwards them via the ``config`` dict to ``install``.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

from ...schema import Edge, Event

MANIFEST = ExtensionManifest(
    name="auditor_get_event_detail_tool",
    description=(
        "Register the get_event_detail([ids]) drill-down tool for the "
        "Phase 2 auditor child session. Returns the full Event.to_dict() "
        "plus outgoing+incoming Edge.to_dict() records for each requested "
        "event id. Used in degraded-prompt mode when the prompt only "
        "carries summarised events."
    ),
    registers=("tool:get_event_detail",),
    config_schema={
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {"type": "object"},
            },
            "edges": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)

GET_EVENT_DETAIL_TOOL_NAME = "get_event_detail"

_GET_EVENT_DETAIL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "description": (
                "Event ids to fetch full details for. Batched to amortize round trips."
            ),
        },
    },
    "required": ["event_ids"],
    "additionalProperties": False,
}

_GET_EVENT_DETAIL_DESCRIPTION = (
    "Fetch the full Event.to_dict() plus outgoing and incoming "
    "Edge.to_dict() lists for every event id in event_ids. Use in "
    "degraded-prompt mode (more events than audit_summary_threshold) "
    "to recover witness fields. Unknown ids appear in a top-level "
    '"missing" list. Out-of-shape input returns a structured '
    "tool-result error rather than raising."
)


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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Register the get_event_detail tool, closing over events/edges."""
    events = _coerce_events(config.get("events", ()))
    edges = _coerce_edges(config.get("edges", ()))

    events_by_id: dict[int, Event] = {ev.id: ev for ev in events}
    outgoing: dict[int, list[Edge]] = {}
    incoming: dict[int, list[Edge]] = {}
    for ed in edges:
        outgoing.setdefault(ed.src, []).append(ed)
        incoming.setdefault(ed.dst, []).append(ed)

    async def _get_event_detail(args: dict[str, Any]) -> ToolResult:
        ids_raw = args.get("event_ids")
        if not isinstance(ids_raw, list) or not ids_raw:
            return _err("get_event_detail rejected: event_ids must be a non-empty list of integers")
        for x in ids_raw:
            # bool is a subclass of int — exclude.
            if not isinstance(x, int) or isinstance(x, bool):
                return _err(
                    "get_event_detail rejected: every event_ids entry must "
                    f"be an integer, got {type(x).__name__!r}"
                )
            if x < 0:
                return _err(f"get_event_detail rejected: negative id {x} is invalid")

        result: dict[str, Any] = {}
        missing: list[int] = []
        for eid in ids_raw:
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

    api.register_tool(
        FunctionTool(
            name=GET_EVENT_DETAIL_TOOL_NAME,
            description=_GET_EVENT_DETAIL_DESCRIPTION,
            parameters=_GET_EVENT_DETAIL_PARAMETERS,
            fn=_get_event_detail,
        )
    )


__all__ = ["GET_EVENT_DETAIL_TOOL_NAME", "MANIFEST", "install"]
