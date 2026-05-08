"""§11 single-file extension: register the ``submit_audit`` terminal tool.

The diagnostic-agent's audit output rides on this tool's ``arguments`` —
a structured JSON Schema-validated payload — instead of being scraped out
of free-form trailing text. The agent calls ``submit_audit(events=...,
verdict=...)``; the kernel records the call as a :class:`ToolCallBlock`
in the assistant message stream, executes the tool, sees the returned
:class:`ToolTerminate`, and ends the child loop.

The adapter (:mod:`llmharness.adapters.agentm`) reads the structured
payload directly off the last assistant message's :class:`ToolCallBlock`
arguments — no JSON extraction, no balanced-brace scanning, no fenced
``json`` block heuristics.

Tier 1, api_version 1, single tool registration.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="submit_tool",
    description=(
        "Register the submit_audit terminal tool — the diagnostic agent calls "
        "it once with structured (events, verdict) and the loop ends."
    ),
    registers=("tool:submit_audit",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


SUBMIT_AUDIT_TOOL_NAME = "submit_audit"

# JSON Schema mirrors the typed payloads in ``llmharness.schema``. Keeping
# this schema explicit here (rather than auto-deriving) lets us tune the
# field descriptions for the LLM provider's tool-use surface — model-side
# schema validation is the whole point of moving from text-JSON to a tool
# call, so the descriptions matter as much as the types.
_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "task",
                "hypothesis",
                "evidence",
                "decision",
                "action",
                "reflection",
                "conclusion",
            ],
            "description": "Closed-set event kind. Anything outside the enum is dropped.",
        },
        "summary": {
            "type": "string",
            "description": "One short sentence describing the event.",
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

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "drift": {
            "type": "boolean",
            "description": "True iff a concrete cognitive drift has been identified.",
        },
        "type": {
            "type": ["string", "null"],
            "enum": [
                "task_drift",
                "evidence_ignored",
                "premature_conclusion",
                "stuck_loop",
                None,
            ],
            "description": (
                "Drift category when drift=true; null otherwise. Pick the "
                "closest of the four types."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Self-reported certainty in [0, 1].",
        },
        "reminder": {
            "type": "string",
            "description": (
                "Free-text advisory body to inject on the next user prompt. "
                "Empty string when drift=false. Do NOT prepend '[harness] '."
            ),
        },
        "matched_event_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Ids of events the verdict attaches to.",
        },
        "cited_cards": {
            "type": "array",
            "items": {"type": "string"},
            "description": "AFC card ids that materially shaped the finding.",
        },
        "downstream_reaction": {
            "type": ["string", "null"],
            "description": (
                "Free-text note about whether the prior reminder in "
                "recent_alerts was heeded. Null only on the first audit "
                "firing of a session."
            ),
        },
    },
    "required": ["drift"],
    "additionalProperties": False,
}

SUBMIT_AUDIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": _EVENT_SCHEMA,
            "description": (
                "Stage-A new events extracted from the trajectory. Empty "
                "array is valid when self-silenced or when no semantically "
                "meaningful new move occurred."
            ),
        },
        "verdict": _VERDICT_SCHEMA,
    },
    "required": ["events", "verdict"],
    "additionalProperties": False,
}

_SUBMIT_AUDIT_DESCRIPTION = (
    "Submit the cognitive-audit output. Call this exactly ONCE per audit "
    "firing as your final action — the loop terminates the moment this "
    "tool returns. Pass the new events extracted in stage A and the verdict "
    "produced in stage B. To stay silent, pass verdict={drift:false} and "
    "events=[]."
)

_TERMINATE_REASON = "llmharness:submit_audit"


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config  # no knobs in V0

    async def _submit(args: dict[str, Any]) -> ToolTerminate:
        del args  # adapter reads args off the ToolCallBlock; no echo needed
        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(type="text", text="audit submitted")]
            ),
            reason=_TERMINATE_REASON,
        )

    api.register_tool(
        FunctionTool(
            name=SUBMIT_AUDIT_TOOL_NAME,
            description=_SUBMIT_AUDIT_DESCRIPTION,
            parameters=SUBMIT_AUDIT_PARAMETERS,
            fn=_submit,
        )
    )


__all__ = [
    "MANIFEST",
    "SUBMIT_AUDIT_PARAMETERS",
    "SUBMIT_AUDIT_TOOL_NAME",
    "install",
]
