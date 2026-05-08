"""§11 single-file extension: register the ``submit_verdict`` terminal tool.

The auditor child session terminates by calling ``submit_verdict(verdict=...)``.
The kernel records the call as a :class:`ToolCallBlock`, executes the tool,
sees the returned :class:`ToolTerminate`, and ends the loop. The adapter
reads the structured payload directly off the assistant message — no JSON
extraction from free text.

The verdict schema includes a JSON Schema ``if/then`` block that enforces
``drift == true ⇒ type required AND non-null``. This closes the V0 bug
where a model could emit ``drift=true, type=null`` and the adapter would
silently drop the reminder. With ``if/then`` declared at the verdict level,
the provider-side schema validation rejects the malformed call before it
reaches the adapter — so a non-conforming auditor surfaces as
``audit_no_call`` (a visible failure) rather than a silent no-op.

The drift-type enum is sourced from
:data:`llmharness.audit._enum_schema.DRIFT_TYPE_VALUES` so the schema and
:class:`llmharness.schema.DriftType` cannot drift apart silently.

Tier 1, api_version 1, single tool registration.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from .._enum_schema import DRIFT_TYPE_VALUES

MANIFEST = ExtensionManifest(
    name="auditor_submit_tool",
    description=(
        "Register the submit_verdict terminal tool — the Phase 2 auditor "
        "calls it once with the structured verdict and the loop ends."
    ),
    registers=("tool:submit_verdict",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


SUBMIT_VERDICT_TOOL_NAME = "submit_verdict"


# The drift-type enum value list contains string members plus a trailing
# None. JSON Schema accepts ``null`` in ``enum`` directly, so we forward
# the list as-is. The ``type`` field declares ``["string", "null"]`` so
# Python ``None`` lands as JSON ``null`` on the wire.
_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "drift": {
            "type": "boolean",
            "description": "True iff a concrete cognitive drift has been identified.",
        },
        "type": {
            "type": ["string", "null"],
            "enum": DRIFT_TYPE_VALUES,
            "description": (
                "Drift category when drift=true; null otherwise. Pick the "
                "closest enum member."
            ),
        },
        "reminder": {
            "type": ["object", "null"],
            "description": (
                "Structured advisory payload to inject on the next user "
                "prompt, or null when drift=false. Do NOT prepend "
                "'[harness] ' — the adapter handles that."
            ),
        },
        "cited_cards": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": (
                "AFC card ids that materially shaped the finding. Free-text "
                "ids; null when no card was decisive."
            ),
        },
        "downstream_reaction": {
            "type": ["string", "null"],
            "description": (
                "Free-text note about whether the prior reminder in "
                "recent_verdicts was heeded. Null only on the first "
                "auditor firing of a session."
            ),
        },
    },
    "required": ["drift"],
    # Closes V0's "drift=true with type=null silently dropped" bug at the
    # provider edge. With this clause, the provider-side schema validator
    # rejects the call before it reaches the adapter — a non-conforming
    # auditor surfaces as audit_no_call rather than a silent no-op.
    "if": {
        "properties": {"drift": {"const": True}},
        "required": ["drift"],
    },
    "then": {
        "required": ["type"],
        "properties": {
            "type": {
                "type": "string",
                "enum": [v for v in DRIFT_TYPE_VALUES if v is not None],
            },
        },
    },
    "additionalProperties": False,
}


SUBMIT_VERDICT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": _VERDICT_SCHEMA,
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


_SUBMIT_VERDICT_DESCRIPTION = (
    "Submit the cognitive-audit verdict. Call this exactly ONCE per "
    "auditor firing as your final action — the loop terminates the moment "
    "this tool returns. To stay silent, pass verdict={drift:false, type:null}."
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config  # no knobs in V1

    async def _submit(args: dict[str, Any]) -> ToolTerminate:
        del args  # adapter reads args off the ToolCallBlock; no echo needed
        return ToolTerminate(
            result=ToolResult(
                content=[TextContent(type="text", text="verdict submitted")]
            ),
            reason="llmharness:submit_verdict",
        )

    api.register_tool(
        FunctionTool(
            name=SUBMIT_VERDICT_TOOL_NAME,
            description=_SUBMIT_VERDICT_DESCRIPTION,
            parameters=SUBMIT_VERDICT_PARAMETERS,
            fn=_submit,
        )
    )


__all__ = [
    "MANIFEST",
    "SUBMIT_VERDICT_PARAMETERS",
    "SUBMIT_VERDICT_TOOL_NAME",
    "install",
]
