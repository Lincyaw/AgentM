"""Build the v3 extractor tool trio bound to a per-firing ExtractionState.

Three tools (design §7.1):

- ``register_event(turn_indices, kind, summary)`` — append an event,
  return its monotonic id.
- ``add_edge(src_event_id, dst_event_id, kind, reason, src_turns,
  dst_turns, cited_entities=[], cited_quote="")`` — validate witness +
  cycle + turn-subset rules, append on pass, return error otherwise.
  Retry budget per ``(src, dst, kind)`` tuple is enforced inside the
  state object.
- ``submit_extraction()`` — terminal tool, takes no arguments, ends the
  child loop with ``ToolTerminate`` reason
  ``"llmharness:submit_extraction"``.

The adapter constructs the ``ExtractionState``, calls
``build_extractor_tools(state)``, and registers the resulting tools on
the child session. The state IS the output: the adapter calls
``state.freeze()`` after the child loop terminates rather than parsing
JSON out of any tool-call argument.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate

from ...schema import EdgeKind, EventKind
from .._enum_schema import EDGE_KIND_VALUES, EVENT_KIND_VALUES
from .state import ExtractionState

REGISTER_EVENT_TOOL_NAME = "register_event"
ADD_EDGE_TOOL_NAME = "add_edge"
SUBMIT_EXTRACTION_TOOL_NAME = "submit_extraction"
SUBMIT_EXTRACTION_REASON = "llmharness:submit_extraction"

EXTRACTOR_TOOL_NAMES: tuple[str, str, str] = (
    REGISTER_EVENT_TOOL_NAME,
    ADD_EDGE_TOOL_NAME,
    SUBMIT_EXTRACTION_TOOL_NAME,
)

_REGISTER_EVENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn_indices": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Trajectory indices this event was extracted from. Must "
                "be non-empty; used as the allowed source-turn set for "
                "any edge whose endpoint is this event."
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
    },
    "required": ["turn_indices", "kind", "summary"],
    "additionalProperties": False,
}

_ADD_EDGE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "src_event_id": {"type": "integer"},
        "dst_event_id": {"type": "integer"},
        "kind": {
            "type": "string",
            "enum": list(EDGE_KIND_VALUES),
            "description": (
                "'data' for content/data flow (cited_entities required); "
                "'ref' for referential mention (cited_quote required)."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining the connection.",
        },
        "src_turns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Subset of events[src].source_turns to witness against.",
        },
        "dst_turns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Subset of events[dst].source_turns to witness against.",
        },
        "cited_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Required non-empty when kind='data'. Each entity must "
                "appear (case+ws normalized substring) in BOTH the "
                "concatenated src_turns text and dst_turns text."
            ),
        },
        "cited_quote": {
            "type": "string",
            "description": (
                "Required non-empty when kind='ref'. Must appear "
                "(case+ws normalized substring) in BOTH src_turns text "
                "and dst_turns text."
            ),
        },
    },
    "required": [
        "src_event_id",
        "dst_event_id",
        "kind",
        "reason",
        "src_turns",
        "dst_turns",
    ],
    "additionalProperties": False,
}

_SUBMIT_EXTRACTION_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _ok(payload: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=payload)])


def _err(payload: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=payload)], is_error=True)


def build_extractor_tools(state: ExtractionState) -> list[FunctionTool]:
    """Build the three v3 extractor tools, each closing over ``state``."""

    async def _register_event(args: dict[str, Any]) -> ToolResult:
        turn_indices_raw = args.get("turn_indices")
        kind_raw = args.get("kind")
        summary_raw = args.get("summary")

        if not isinstance(turn_indices_raw, list) or not turn_indices_raw:
            return _err("register_event: 'turn_indices' must be a non-empty array of integers")
        turn_indices: list[int] = []
        for item in turn_indices_raw:
            if isinstance(item, bool) or not isinstance(item, int):
                return _err(f"register_event: 'turn_indices' contains non-integer entry {item!r}")
            turn_indices.append(item)
        if not isinstance(kind_raw, str):
            return _err("register_event: 'kind' must be a string")
        try:
            kind = EventKind(kind_raw)
        except ValueError:
            return _err(f"register_event: 'kind' {kind_raw!r} not in {EVENT_KIND_VALUES}")
        if not isinstance(summary_raw, str) or not summary_raw.strip():
            return _err("register_event: 'summary' must be a non-empty string")

        event_id = state.register_event(
            kind=kind,
            summary=summary_raw,
            source_turns=turn_indices,
        )
        return _ok(f'{{"event_id": {event_id}}}')

    async def _add_edge(args: dict[str, Any]) -> ToolResult:
        src_raw = args.get("src_event_id")
        dst_raw = args.get("dst_event_id")
        kind_raw = args.get("kind")
        reason_raw = args.get("reason")
        src_turns_raw = args.get("src_turns")
        dst_turns_raw = args.get("dst_turns")
        cited_entities_raw = args.get("cited_entities", [])
        cited_quote_raw = args.get("cited_quote", "")

        if isinstance(src_raw, bool) or not isinstance(src_raw, int):
            return _err("add_edge: 'src_event_id' must be an integer")
        if isinstance(dst_raw, bool) or not isinstance(dst_raw, int):
            return _err("add_edge: 'dst_event_id' must be an integer")
        if not isinstance(kind_raw, str):
            return _err("add_edge: 'kind' must be a string")
        try:
            kind = EdgeKind(kind_raw)
        except ValueError:
            return _err(f"add_edge: 'kind' {kind_raw!r} not in {EDGE_KIND_VALUES}")
        if not isinstance(reason_raw, str):
            return _err("add_edge: 'reason' must be a string")
        if not isinstance(src_turns_raw, list):
            return _err("add_edge: 'src_turns' must be an array of integers")
        if not isinstance(dst_turns_raw, list):
            return _err("add_edge: 'dst_turns' must be an array of integers")
        src_turns: list[int] = []
        for item in src_turns_raw:
            if isinstance(item, bool) or not isinstance(item, int):
                return _err(f"add_edge: 'src_turns' contains non-integer entry {item!r}")
            src_turns.append(item)
        dst_turns: list[int] = []
        for item in dst_turns_raw:
            if isinstance(item, bool) or not isinstance(item, int):
                return _err(f"add_edge: 'dst_turns' contains non-integer entry {item!r}")
            dst_turns.append(item)
        if not isinstance(cited_entities_raw, list):
            return _err("add_edge: 'cited_entities' must be an array of strings")
        cited_entities: list[str] = []
        for item in cited_entities_raw:
            if not isinstance(item, str):
                return _err(f"add_edge: 'cited_entities' contains non-string entry {item!r}")
            cited_entities.append(item)
        if not isinstance(cited_quote_raw, str):
            return _err("add_edge: 'cited_quote' must be a string")

        err = state.add_edge(
            src_event_id=src_raw,
            dst_event_id=dst_raw,
            kind=kind,
            reason=reason_raw,
            src_turns=src_turns,
            dst_turns=dst_turns,
            cited_entities=cited_entities,
            cited_quote=cited_quote_raw,
        )
        if err is None:
            return _ok('{"ok": true}')
        return _err(err)

    async def _submit_extraction(args: dict[str, Any]) -> ToolTerminate:
        del args  # state IS the output
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="extraction submitted")]),
            reason=SUBMIT_EXTRACTION_REASON,
        )

    return [
        FunctionTool(
            name=REGISTER_EVENT_TOOL_NAME,
            description=(
                "Register a new semantic event extracted from one or more "
                "turns. Returns its monotonic event_id, which you then use "
                "as src_event_id / dst_event_id when calling add_edge."
            ),
            parameters=_REGISTER_EVENT_PARAMETERS,
            fn=_register_event,
        ),
        FunctionTool(
            name=ADD_EDGE_TOOL_NAME,
            description=(
                "Connect two registered events. Provide a witness "
                "(cited_entities for kind='data', cited_quote for "
                "kind='ref'); the harness verifies the witness against "
                "the actual turn texts. Up to 3 attempts per "
                "(src_event_id, dst_event_id, kind); the 3rd failure "
                "drops the edge with the terminal error 'giving up on "
                "this edge'."
            ),
            parameters=_ADD_EDGE_PARAMETERS,
            fn=_add_edge,
        ),
        FunctionTool(
            name=SUBMIT_EXTRACTION_TOOL_NAME,
            description=(
                "Terminator. Call this exactly ONCE as your final action "
                "to end the extractor loop after you have registered all "
                "events and edges. Takes no arguments — the harness reads "
                "events and edges from the registered state."
            ),
            parameters=_SUBMIT_EXTRACTION_PARAMETERS,
            fn=_submit_extraction,
        ),
    ]


__all__ = [
    "ADD_EDGE_TOOL_NAME",
    "EXTRACTOR_TOOL_NAMES",
    "REGISTER_EVENT_TOOL_NAME",
    "SUBMIT_EXTRACTION_REASON",
    "SUBMIT_EXTRACTION_TOOL_NAME",
    "build_extractor_tools",
]
