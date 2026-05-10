"""§11 single-file extension: register the ``get_turn(idx)`` drill-down tool.

The auditor child session can call ``get_turn(idx)`` to pull the serialized
raw conversation turn at absolute index ``idx`` from the parent trajectory
snapshot captured at spawn time (design §6.3).

Out-of-range or non-integer ``idx`` returns a structured ``ToolResult`` with
``is_error=True`` rather than raising — the auditor loop must never crash
from a bad index query.

The snapshot is a pre-serialized ``list[dict[str, Any]]`` produced by
``_serialize_full_trajectory`` in ``adapters/agentm.py`` and forwarded
through ``compose_auditor_extensions(trajectory_snapshot=...)``.  It is
captured once at ``TurnEndEvent`` time so the auditor sees a consistent
view of the trajectory throughout a single firing.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="auditor_get_turn_tool",
    description=(
        "Register the get_turn(idx) drill-down tool for the Phase 2 auditor "
        "child session. Returns the serialized parent trajectory turn at "
        "index idx, or a structured error tool-result for out-of-range / "
        "non-integer idx."
    ),
    registers=("tool:get_turn",),
    config_schema={
        "type": "object",
        "properties": {
            "trajectory_snapshot": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)

GET_TURN_TOOL_NAME = "get_turn"

_GET_TURN_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "idx": {
            "type": "integer",
            "description": (
                "Absolute index of the message in the parent trajectory "
                "(0-based). Use the source_turns value from an audit event."
            ),
        },
    },
    "required": ["idx"],
    "additionalProperties": False,
}

_GET_TURN_DESCRIPTION = (
    "Fetch the serialized raw trajectory turn at index ``idx`` from the "
    "parent session. Use to verify an event's source_turns reference "
    "against the actual trajectory text. "
    "Out-of-range or invalid idx returns a structured tool-result error "
    "rather than crashing the auditor loop."
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Register the get_turn tool, closing over the trajectory snapshot."""
    snapshot_raw = config.get("trajectory_snapshot", [])
    snapshot: list[dict[str, Any]] = list(snapshot_raw) if isinstance(snapshot_raw, list) else []

    async def _get_turn(args: dict[str, Any]) -> ToolResult:
        idx_raw = args.get("idx")
        # Reject non-int and bool (bool is a subclass of int in Python).
        if not isinstance(idx_raw, int) or isinstance(idx_raw, bool):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"get_turn rejected: idx must be an integer, "
                            f"got {type(idx_raw).__name__!r}"
                        ),
                    )
                ],
                is_error=True,
            )
        if idx_raw < 0 or idx_raw >= len(snapshot):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"get_turn rejected: idx {idx_raw} out of range [0, {len(snapshot)})"
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(snapshot[idx_raw], ensure_ascii=False),
                )
            ],
            is_error=False,
        )

    api.register_tool(
        FunctionTool(
            name=GET_TURN_TOOL_NAME,
            description=_GET_TURN_DESCRIPTION,
            parameters=_GET_TURN_PARAMETERS,
            fn=_get_turn,
        )
    )


__all__ = ["GET_TURN_TOOL_NAME", "MANIFEST", "install"]
