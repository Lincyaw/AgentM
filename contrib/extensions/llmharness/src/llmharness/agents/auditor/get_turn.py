"""``get_turn(idx)`` drill-down tool — Pydantic-backed.

The auditor child session can call ``get_turn(idx)`` to pull the serialized
raw conversation turn at absolute index ``idx`` from the parent trajectory
snapshot captured at spawn time (design §6.3). Out-of-range or non-integer
``idx`` returns a structured ``ToolResult`` with ``is_error=True`` rather
than raising — the auditor loop must never crash from a bad index query.

The snapshot is a pre-serialized ``list[dict[str, Any]]`` produced by
``_serialize_full_trajectory`` in ``adapters/agentm.py`` and forwarded
through ``compose_auditor_extensions(trajectory_snapshot=...)``. It is
captured once at ``TurnEndEvent`` time so the auditor sees a consistent
view of the trajectory throughout a single firing.

This module is **not** an atom — the merged ``atom.py`` calls
:func:`build_get_turn_tool` to mint a stateful :class:`FunctionTool` over
the configured snapshot at install time.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.lib import pydantic_to_tool_schema
from pydantic import BaseModel, ConfigDict, Field

from llmharness.runtime.decorator import harness_tool

GET_TURN_TOOL_NAME = "get_turn"


class GetTurnArgs(BaseModel):
    # No class docstring — pydantic would emit it as a top-level schema
    # ``description`` that the hand-written V1 schema did not carry.
    model_config = ConfigDict(extra="forbid", strict=True)

    idx: int = Field(
        description=(
            "Absolute index of the message in the parent trajectory "
            "(0-based). Use the source_turns value from an audit event."
        ),
    )


# Stateless schema constant — exported for downstream training code that
# needs to register the tool surface without an actual snapshot in hand.
GET_TURN_PARAMETERS: dict[str, Any] = pydantic_to_tool_schema(GetTurnArgs)


def build_get_turn_tool(snapshot: list[dict[str, Any]]) -> FunctionTool:
    """Mint a :class:`FunctionTool` closing over ``snapshot``."""

    @harness_tool(GET_TURN_TOOL_NAME)
    async def _get_turn(args: GetTurnArgs, _ctx: Any) -> ToolResult:
        """Fetch the serialized raw trajectory turn at index ``idx`` from the parent session. Use to verify an event's source_turns reference against the actual trajectory text. Out-of-range or invalid idx returns a structured tool-result error rather than crashing the auditor loop."""
        # Pydantic accepts bools as ints (bool is a subclass of int). Reject
        # explicitly to preserve the V1 boundary semantics: only true ints.
        if isinstance(args.idx, bool):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="get_turn rejected: idx must be an integer, got 'bool'",
                    )
                ],
                is_error=True,
            )
        if args.idx < 0 or args.idx >= len(snapshot):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"get_turn rejected: idx {args.idx} out of range [0, {len(snapshot)})"
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(snapshot[args.idx], ensure_ascii=False),
                )
            ],
            is_error=False,
        )

    return _get_turn


__all__ = [
    "GET_TURN_PARAMETERS",
    "GET_TURN_TOOL_NAME",
    "GetTurnArgs",
    "build_get_turn_tool",
]
