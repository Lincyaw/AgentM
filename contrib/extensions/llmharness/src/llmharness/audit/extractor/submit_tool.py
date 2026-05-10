"""§11 single-file extension: register the ``submit_events`` terminal tool.

The extractor child session emits its structured event list via this
tool's ``arguments`` — a JSON-Schema-validated payload — instead of via
trailing free-form text. The agent calls ``submit_events(events=[...])``;
the kernel records the call as a :class:`ToolCallBlock`, executes the
tool, sees the returned :class:`ToolTerminate`, and ends the child loop.

The adapter (:mod:`llmharness.adapters.agentm`) reads the structured
payload directly off the last assistant message's :class:`ToolCallBlock`
arguments — no JSON extraction, no fenced-block heuristics.

Tier 1, api_version 1, single tool registration.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from .._enum_schema import EVENT_KIND_VALUES

MANIFEST = ExtensionManifest(
    name="extractor_submit_tool",
    description=(
        "Register the submit_events terminal tool — the extractor child "
        "calls it once with a structured events list and the loop ends."
    ),
    registers=("tool:submit_events",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


SUBMIT_EVENTS_TOOL_NAME = "submit_events"

# JSON Schema mirrors the typed payloads in ``llmharness.schema``. The
# ``kind`` enum is derived from ``EVENT_KIND_VALUES`` rather than hand
# listed so this stays in lockstep with ``EventKind`` in ``schema.py``.
_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": list(EVENT_KIND_VALUES),
            "description": "Closed-set event kind. Anything outside the enum is dropped.",
        },
        "summary": {
            "type": "string",
            "description": (
                "One short sentence describing the event. When this event "
                "contradicts or refines a referenced prior event, embed the "
                "relation in free text (no preset edge-type vocabulary)."
            ),
        },
        "source_turns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Trajectory indices this event was extracted from.",
        },
        "refs": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Prior event ids this event references.",
        },
    },
    "required": ["kind", "summary"],
    "additionalProperties": False,
}


SUBMIT_EVENTS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": _EVENT_SCHEMA,
            "description": (
                "New events extracted from the new-turn slice. An empty "
                "array is legal — the adapter classifies it as "
                "``extractor_empty`` when the input window was non-trivial."
            ),
        },
    },
    "required": ["events"],
    "additionalProperties": False,
}


_SUBMIT_EVENTS_DESCRIPTION = (
    "Submit the extracted event list. Call this exactly ONCE per extractor "
    "firing as your final action — the loop terminates the moment this "
    "tool returns. Pass the list of NEW events extracted from the new-turn "
    "slice; pass an empty array if no semantically meaningful new move "
    "occurred."
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config  # no knobs in V1

    async def _submit(args: dict[str, Any]) -> ToolTerminate:
        del args  # adapter reads args off the ToolCallBlock; no echo needed
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text="events submitted")]),
            reason="llmharness:submit_events",
        )

    api.register_tool(
        FunctionTool(
            name=SUBMIT_EVENTS_TOOL_NAME,
            description=_SUBMIT_EVENTS_DESCRIPTION,
            parameters=SUBMIT_EVENTS_PARAMETERS,
            fn=_submit,
        )
    )


__all__ = [
    "MANIFEST",
    "SUBMIT_EVENTS_PARAMETERS",
    "SUBMIT_EVENTS_TOOL_NAME",
    "install",
]
