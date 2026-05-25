"""``delete_node`` extractor tool — Pydantic-backed.

Delete one event node from the current folded graph; all incident
edges cascade off at fold time. Used both to merge duplicates emitted
by prior firings and to clear a passthrough node so a wider linear
block can be re-emitted.

This module is **not** an atom.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, ConfigDict, Field

from ...toolkit.decorator import harness_tool
from ...validation.witness_errors import format_witness_error
from ..state import ExtractionState
from ..state_echo import state_echo

DELETE_NODE_TOOL_NAME = "delete_node"


class DeleteNodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "Event node id to delete. May reference a node from this "
            "firing or from any prior firing — the resulting "
            "NodeDelete cascades to all incident edges at fold time."
        ),
    )


def build_delete_node_tool(state: ExtractionState) -> FunctionTool:
    @harness_tool(DELETE_NODE_TOOL_NAME)
    async def _delete_node(args: DeleteNodeArgs, _ctx: Any) -> ToolResult:
        """Delete one event node from the current graph (this firing or any prior firing). Every edge incident to that node is removed automatically at fold time."""
        result = state.apply_node_delete(args.id)
        if isinstance(result, str):
            nodes, _edges = state._folded_view()
            sample_ids = sorted(nodes.keys())[:5]
            options = [
                f"re-call delete_node with one of the existing ids: {sample_ids}",
                "if you meant to skip this delete, just proceed with upsert_node / upsert_edge",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=f"delete_node(id={args.id})",
                            state_echo=state_echo(state),
                            options=options,
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        )

    return _delete_node


__all__ = ["DELETE_NODE_TOOL_NAME", "DeleteNodeArgs", "build_delete_node_tool"]
