"""Shared helper for locating a spawned child session by trace label."""
from __future__ import annotations

from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext


def find_child_session(
    ctx: WorkflowContext,
    label: str,
) -> dict[str, Any] | None:
    """Return the most recent child session whose label matches, if any.

    Used to feed an agent's own investigation transcript (its tool calls)
    into the gate that reviews it for completeness.
    """
    for child in reversed(ctx.child_sessions):
        if child.get("trace_label") == label or child.get("workflow_node_id") == label:
            return dict(child)
    return None
