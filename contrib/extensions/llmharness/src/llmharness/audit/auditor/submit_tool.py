"""§11 single-file extension: register the ``submit_verdict`` terminal tool (V2).

The auditor child session terminates by calling ``submit_verdict(verdict=...)``.
The kernel records the call as a :class:`ToolCallBlock`, executes the tool,
sees the returned :class:`ToolTerminate`, and ends the loop. The adapter
reads the structured payload directly off the assistant message — no JSON
extraction from free text.

V2 schema (design §6.2): ``surface_reminder``, ``reminder_text``,
``continuation_notes``, ``matched_event_ids``, ``cited_cards``.
No ``drift`` field, no ``DriftType`` enum, no ``if/then`` clause.
The ``surface_reminder=True ⇒ non-empty reminder_text`` invariant is
enforced at the adapter-side coercer (:class:`RawVerdictOutput`), not at
the JSON schema level, so the LLM sees a friendly retry error rather than
a provider-side rejection that may be invisible in the tool result.

Tier 1, api_version 1, single tool registration.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

from .output import AuditorOutputError, RawVerdictOutput

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


# V2 verdict schema (design §6.2).
# No if/then: provider-side schema complexity is avoided; the adapter-side
# coercer (RawVerdictOutput.from_dict) enforces surface_reminder=True ⇒
# non-empty reminder_text and returns a visible tool-result error on
# violation so the auditor LLM can retry.
_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "surface_reminder": {
            "type": "boolean",
            "description": (
                "True if a concrete concern warrants surfacing a reminder to "
                "the main agent before its next turn. False for a silent verdict."
            ),
        },
        "reminder_text": {
            "type": "string",
            "description": (
                "The advisory the main agent will read on its next turn. "
                "Must be non-empty when surface_reminder=true; empty string "
                "when surface_reminder=false. Do NOT prepend '[harness] ' — "
                "the adapter handles that."
            ),
        },
        "continuation_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Free-text notes to yourself across firings — what you asked "
                "yourself to recheck next time. May be empty. These are "
                "forwarded into your context at the next auditor firing as "
                "'continuation_notes_from_prior_firing'."
            ),
        },
        "matched_event_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "IDs of the audit graph events that materially supported "
                "this verdict. Non-empty when surface_reminder=true so the "
                "finding is traceable; may be empty when surface_reminder=false."
            ),
        },
        "cited_cards": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "AFC card ids that materially shaped the finding. Empty "
                "array when no card was decisive."
            ),
        },
    },
    "required": [
        "surface_reminder",
        "reminder_text",
        "continuation_notes",
        "matched_event_ids",
        "cited_cards",
    ],
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
    "this tool returns. To stay silent, pass "
    'verdict={surface_reminder:false, reminder_text:"", '
    "continuation_notes:[], matched_event_ids:[], cited_cards:[]}."
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config  # no knobs in V1

    async def _submit(args: dict[str, Any]) -> ToolTerminate | ToolResult:
        # Run the adapter-side coercion here so a malformed payload becomes
        # a visible tool-result error inside the auditor child loop. The
        # LLM sees the error message and gets another turn to retry; only
        # on a well-shaped payload do we ToolTerminate the child loop.
        try:
            RawVerdictOutput.from_dict(args).to_verdict()
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
