"""``delete_edge`` extractor tool — Pydantic-backed.

Delete one edge identified by ``(src, dst, kind)``. ``kind`` is
mandatory: the same ``(src, dst)`` pair may carry both a ``data`` and a
``ref`` edge in the op log, so ``(src, dst)`` alone is ambiguous.

This module is **not** an atom.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, ConfigDict, Field

from llmharness.audit.toolkit.decorator import harness_tool
from llmharness.audit.validation.witness_errors import format_witness_error

from ..state import ExtractionState
from ..state_echo import state_echo

DELETE_EDGE_TOOL_NAME = "delete_edge"

_EdgeKindLiteral = Literal["data", "ref"]


class DeleteEdgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int = Field(description="Source node id of the edge to delete.")
    dst: int = Field(description="Destination node id of the edge to delete.")
    kind: _EdgeKindLiteral = Field(
        description=(
            "Required: the edge kind to delete. Without this the "
            "selector is ambiguous — the same (src, dst) pair may "
            "carry both a 'data' and a 'ref' edge in the persistent "
            "op log."
        ),
    )


def build_delete_edge_tool(state: ExtractionState) -> FunctionTool:
    @harness_tool(DELETE_EDGE_TOOL_NAME)
    async def _delete_edge(args: DeleteEdgeArgs, _ctx: Any) -> ToolResult:
        """Delete one edge identified by (src, dst, kind). 'kind' is mandatory because the same (src, dst) pair may carry both a 'data' and a 'ref' edge."""
        raw = args.model_dump()
        result = state.apply_edge_delete(raw)
        if isinstance(result, str):
            _nodes, edges = state._folded_view()
            present = sorted(edges.keys())[:8]
            options = [
                f"re-call delete_edge with an existing (src, dst, kind) "
                f"triple — folded view contains {present}",
                "if you wanted to remove a node entirely, call delete_node(<id>) "
                "instead and let the edge cascade off automatically",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=f"delete_edge(src={args.src}, dst={args.dst}, kind={args.kind})",
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

    return _delete_edge


__all__ = ["DELETE_EDGE_TOOL_NAME", "DeleteEdgeArgs", "build_delete_edge_tool"]
