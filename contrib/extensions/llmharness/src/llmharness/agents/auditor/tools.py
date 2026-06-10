"""Auditor tool surface: submit_verdict (terminal) + optional drill-down tools.

The atom registers tools selected by the parent's config. submit_verdict
terminates the child loop; get_turn and get_event_detail are optional
drill-down tools for degraded-prompt mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmharness.schema import Edge, Event, Verdict

# ---------------------------------------------------------------------------
# Section 2: Output parsing
# ---------------------------------------------------------------------------


class AuditorOutputError(Exception):
    """Raised when the auditor's submit_verdict payload is malformed."""


@dataclass(frozen=True)
class RawVerdictOutput:
    """Typed view over the ``{verdict: {...}}`` payload of submit_verdict."""

    surface_reminder: bool
    reminder_text: str
    continuation_notes: list[str]
    matched_event_ids: list[int]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RawVerdictOutput:
        """Parse the tool-call arguments dict into a typed view."""
        verdict_raw = raw.get("verdict")
        if not isinstance(verdict_raw, dict):
            raise AuditorOutputError("submit_verdict payload missing object-typed 'verdict' field")

        if "surface_reminder" not in verdict_raw:
            raise AuditorOutputError(
                "submit_verdict.verdict missing required 'surface_reminder' field"
            )
        surface_reminder_raw = verdict_raw["surface_reminder"]
        if not isinstance(surface_reminder_raw, bool):
            raise AuditorOutputError(
                f"submit_verdict.verdict.surface_reminder must be bool, "
                f"got {type(surface_reminder_raw).__name__}"
            )

        reminder_text_raw = verdict_raw.get("reminder_text", "")
        if not isinstance(reminder_text_raw, str):
            raise AuditorOutputError("submit_verdict.verdict.reminder_text must be a string")
        if surface_reminder_raw and not reminder_text_raw.strip():
            raise AuditorOutputError(
                "submit_verdict.verdict.reminder_text must be non-empty when surface_reminder=true"
            )

        notes_raw = verdict_raw.get("continuation_notes", [])
        if not isinstance(notes_raw, list) or not all(isinstance(n, str) for n in notes_raw):
            raise AuditorOutputError(
                "submit_verdict.verdict.continuation_notes must be a list of strings"
            )

        matched_raw = verdict_raw.get("matched_event_ids", [])
        if not isinstance(matched_raw, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in matched_raw
        ):
            raise AuditorOutputError(
                "submit_verdict.verdict.matched_event_ids must be a list of integers"
            )

        return cls(
            surface_reminder=surface_reminder_raw,
            reminder_text=reminder_text_raw,
            continuation_notes=list(notes_raw),
            matched_event_ids=list(matched_raw),
        )

    def to_verdict(self) -> Verdict:
        """Materialize a Verdict for the adapter."""
        return Verdict(
            surface_reminder=self.surface_reminder,
            reminder_text=self.reminder_text,
            continuation_notes=list(self.continuation_notes),
            matched_event_ids=list(self.matched_event_ids),
        )


# ---------------------------------------------------------------------------
# Section 3: submit_verdict tool
# ---------------------------------------------------------------------------

SUBMIT_VERDICT_TOOL_NAME = "submit_verdict"


class _VerdictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_reminder: bool = Field(
        description=(
            "True if a concrete concern warrants surfacing a reminder to "
            "the main agent before its next turn. False for a silent verdict."
        ),
    )
    reminder_text: str = Field(
        description=(
            "The advisory the main agent will read on its next turn. "
            "Must be non-empty when surface_reminder=true; empty string "
            "when surface_reminder=false. Do NOT prepend '[harness] ' — "
            "the adapter handles that."
        ),
    )
    continuation_notes: list[str] = Field(
        description=(
            "Free-text notes to yourself across firings — what you asked "
            "yourself to recheck next time. May be empty. These are "
            "forwarded into your context at the next auditor firing as "
            "'continuation_notes_from_prior_firing'."
        ),
    )
    matched_event_ids: list[int] = Field(
        description=(
            "IDs of the audit graph events that materially supported "
            "this verdict. Non-empty when surface_reminder=true so the "
            "finding is traceable; may be empty when surface_reminder=false."
        ),
    )


class SubmitVerdictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: _VerdictModel


async def _submit_handler(args: dict[str, Any]) -> ToolTerminate | ToolResult:
    try:
        parsed = SubmitVerdictArgs.model_validate(args)
    except ValidationError as exc:
        return ToolResult(
            content=[TextContent(type="text", text=f"submit_verdict rejected: {exc}")],
            is_error=True,
        )
    try:
        RawVerdictOutput.from_dict(parsed.model_dump()).to_verdict()
    except AuditorOutputError as exc:
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"submit_verdict rejected: {exc}. "
                        "Reissue submit_verdict with a corrected payload."
                    ),
                )
            ],
            is_error=True,
        )
    return ToolTerminate(
        result=ToolResult(content=[TextContent(type="text", text="verdict submitted")]),
        reason="llmharness:submit_verdict",
    )


SUBMIT_VERDICT_TOOL = FunctionTool(
    name=SUBMIT_VERDICT_TOOL_NAME,
    description="Submit the cognitive-audit verdict. Call exactly ONCE per firing as your final action.",
    parameters=SubmitVerdictArgs,
    fn=_submit_handler,
    metadata={"terminates": True},
)

# ---------------------------------------------------------------------------
# Section 4: get_turn tool
# ---------------------------------------------------------------------------

GET_TURN_TOOL_NAME = "get_turn"


class GetTurnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    idx: int = Field(
        description=(
            "Absolute index of the message in the parent trajectory "
            "(0-based). Use the source_turns value from an audit event."
        ),
    )


GET_TURN_PARAMETERS: dict[str, Any] = pydantic_to_tool_schema(GetTurnArgs)


def build_get_turn_tool(snapshot: list[dict[str, Any]]) -> FunctionTool:
    """Mint a FunctionTool closing over ``snapshot``."""

    async def _get_turn_handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = GetTurnArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"get_turn rejected: {exc}")],
                is_error=True,
            )
        # Pydantic accepts bools as ints; reject explicitly.
        if isinstance(parsed.idx, bool):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="get_turn rejected: idx must be an integer, got 'bool'",
                    )
                ],
                is_error=True,
            )
        if parsed.idx < 0 or parsed.idx >= len(snapshot):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"get_turn rejected: idx {parsed.idx} out of range [0, {len(snapshot)})",
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(snapshot[parsed.idx], ensure_ascii=False),
                )
            ],
            is_error=False,
        )

    return FunctionTool(
        name=GET_TURN_TOOL_NAME,
        description=(
            "Fetch the serialized raw trajectory turn at index idx from the "
            "parent session. Use to verify an event's source_turns reference "
            "against the actual trajectory text."
        ),
        parameters=GetTurnArgs,
        fn=_get_turn_handler,
    )


# ---------------------------------------------------------------------------
# Section 5: get_event_detail tool
# ---------------------------------------------------------------------------

GET_EVENT_DETAIL_TOOL_NAME = "get_event_detail"


class GetEventDetailArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_ids: list[int] = Field(
        min_length=1,
        description="Event ids to fetch full details for. Batched to amortize round trips.",
    )


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
    """Accept either a tuple of Edge or a list of Edge-shaped dicts."""
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


def build_get_event_detail_tool(
    events: tuple[Event, ...] | list[Event] | list[dict[str, Any]],
    edges: tuple[Edge, ...] | list[Edge] | list[dict[str, Any]],
) -> FunctionTool:
    """Mint a FunctionTool closing over ``events`` / ``edges``."""
    events_t = _coerce_events(events)
    edges_t = _coerce_edges(edges)

    events_by_id: dict[int, Event] = {ev.id: ev for ev in events_t}
    outgoing: dict[int, list[Edge]] = {}
    incoming: dict[int, list[Edge]] = {}
    for ed in edges_t:
        outgoing.setdefault(ed.src, []).append(ed)
        incoming.setdefault(ed.dst, []).append(ed)

    async def _get_event_detail_handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = GetEventDetailArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"get_event_detail rejected: {exc}")],
                is_error=True,
            )
        # bool is a subclass of int; strict mode rejects it, but defend the boundary.
        for x in parsed.event_ids:
            if isinstance(x, bool):
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                "get_event_detail rejected: every event_ids entry must "
                                "be an integer, got 'bool'"
                            ),
                        )
                    ],
                    is_error=True,
                )
            if x < 0:
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"get_event_detail rejected: negative id {x} is invalid",
                        )
                    ],
                    is_error=True,
                )

        result: dict[str, Any] = {}
        missing: list[int] = []
        for eid in parsed.event_ids:
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

    return FunctionTool(
        name=GET_EVENT_DETAIL_TOOL_NAME,
        description=(
            "Fetch the full Event plus outgoing and incoming Edge lists for "
            "every event id in event_ids. Use in degraded-prompt mode to "
            "recover witness fields. Unknown ids appear in a top-level "
            "'missing' list."
        ),
        parameters=GetEventDetailArgs,
        fn=_get_event_detail_handler,
    )


# ---------------------------------------------------------------------------
# Section 6: Atom (MANIFEST + install)
# ---------------------------------------------------------------------------

AUDITOR_TOOLS: tuple[FunctionTool, ...] = (SUBMIT_VERDICT_TOOL,)
AUDITOR_TOOL_NAMES: tuple[str, ...] = (
    SUBMIT_VERDICT_TOOL_NAME,
    GET_TURN_TOOL_NAME,
    GET_EVENT_DETAIL_TOOL_NAME,
)
AUDITOR_TERMINATION_REASON: str = "llmharness:submit_verdict"

MANIFEST = ExtensionManifest(
    name="auditor_tools",
    description="Register the auditor tool surface.",
    registers=("tool:submit_verdict", "tool:get_turn", "tool:get_event_detail"),
    config_schema={
        "type": "object",
        "properties": {
            "tools": {"type": "array", "items": {"type": "string"}},
            "trajectory_snapshot": {"type": "array", "items": {"type": "object"}},
            "events": {"type": "array", "items": {"type": "object"}},
            "edges": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    tools_raw = config.get("tools", [SUBMIT_VERDICT_TOOL_NAME])
    tools = list(tools_raw) if isinstance(tools_raw, (list, tuple)) else [SUBMIT_VERDICT_TOOL_NAME]

    unknown = [t for t in tools if t not in AUDITOR_TOOL_NAMES]
    if unknown:
        raise ValueError(
            f"auditor_tools: unknown tool names: {unknown!r}; known: {AUDITOR_TOOL_NAMES!r}"
        )

    if SUBMIT_VERDICT_TOOL_NAME in tools:
        api.register_tool(SUBMIT_VERDICT_TOOL)

    if GET_TURN_TOOL_NAME in tools:
        snapshot = list(config.get("trajectory_snapshot") or [])
        api.register_tool(build_get_turn_tool(snapshot))

    if GET_EVENT_DETAIL_TOOL_NAME in tools:
        api.register_tool(
            build_get_event_detail_tool(config.get("events", ()), config.get("edges", ()))
        )


__all__ = [
    "AUDITOR_TERMINATION_REASON",
    "AUDITOR_TOOLS",
    "AUDITOR_TOOL_NAMES",
    "GET_EVENT_DETAIL_PARAMETERS",
    "GET_EVENT_DETAIL_TOOL_NAME",
    "GET_TURN_PARAMETERS",
    "GET_TURN_TOOL_NAME",
    "MANIFEST",
    "SUBMIT_VERDICT_TOOL",
    "SUBMIT_VERDICT_TOOL_NAME",
    "AuditorOutputError",
    "GetEventDetailArgs",
    "GetTurnArgs",
    "RawVerdictOutput",
    "SubmitVerdictArgs",
    "build_get_event_detail_tool",
    "build_get_turn_tool",
    "install",
]
