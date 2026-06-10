"""``submit_verdict`` terminal tool — Pydantic-backed (auditor V2).

The auditor child session terminates by calling ``submit_verdict(verdict=...)``.
The kernel records the call as a :class:`ToolCallBlock`, executes the tool,
sees the returned :class:`ToolTerminate`, and ends the loop. The adapter
reads the structured payload directly off the assistant message — no JSON
extraction from free text.

Verdict schema (design §6.2): ``surface_reminder``, ``reminder_text``,
``continuation_notes``, ``matched_event_ids``.
No ``drift`` field, no ``DriftType`` enum, no ``if/then`` clause.
The ``surface_reminder=True ⇒ non-empty reminder_text`` invariant is
enforced at the adapter-side coercer (:class:`RawVerdictOutput`), not at
the JSON schema level, so the LLM sees a friendly retry error rather than
a provider-side rejection that may be invisible in the tool result.

This module is **not** an atom — it carries no ``MANIFEST``. The merged
``atom.py`` imports :data:`SUBMIT_VERDICT_TOOL` and registers it alongside
the other auditor tools.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult, ToolTerminate
from pydantic import BaseModel, ConfigDict, Field

from llmharness.audit.toolkit.decorator import harness_tool

from .output import AuditorOutputError, RawVerdictOutput

SUBMIT_VERDICT_TOOL_NAME = "submit_verdict"


class _VerdictModel(BaseModel):
    # No class docstring — pydantic emits it as the schema ``description``,
    # which the hand-written V1 ``_VERDICT_SCHEMA`` did not carry.
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
    # No class docstring + no per-field description on ``verdict`` — matches
    # the hand-written V1 ``SUBMIT_VERDICT_PARAMETERS`` byte-for-byte.
    model_config = ConfigDict(extra="forbid")

    verdict: _VerdictModel


@harness_tool(SUBMIT_VERDICT_TOOL_NAME, terminates=True)
async def _submit(args: SubmitVerdictArgs, _ctx: Any) -> ToolTerminate | ToolResult:
    """Submit the cognitive-audit verdict. Call this exactly ONCE per auditor firing as your final action — the loop terminates the moment this tool returns. To stay silent, pass verdict={surface_reminder:false, reminder_text:"", continuation_notes:[], matched_event_ids:[]}."""
    # Run the adapter-side coercion here so a malformed payload becomes
    # a visible tool-result error inside the auditor child loop. The
    # LLM sees the error message and gets another turn to retry; only
    # on a well-shaped payload do we ToolTerminate the child loop.
    try:
        RawVerdictOutput.from_dict(args.model_dump()).to_verdict()
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


# Exported for the merged atom and downstream training code (rca-autorl).
SUBMIT_VERDICT_TOOL: FunctionTool = _submit


__all__ = [
    "SUBMIT_VERDICT_TOOL",
    "SUBMIT_VERDICT_TOOL_NAME",
    "SubmitVerdictArgs",
]
