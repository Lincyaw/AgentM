"""``reset_extraction`` extractor tool — Pydantic-backed.

Drop all pending events / edges so the model can re-emit the firing's
graph from scratch. The plan is preserved. Used only when the
accumulated graph is unrecoverable by further appends.

This module is **not** an atom.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from pydantic import BaseModel, ConfigDict

from ...toolkit.decorator import harness_tool
from ...validation.witness_errors import format_witness_error
from ..state import ExtractionState
from ..state_echo import state_echo

RESET_EXTRACTION_TOOL_NAME = "reset_extraction"


class ResetExtractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_reset_extraction_tool(state: ExtractionState) -> FunctionTool:
    @harness_tool(RESET_EXTRACTION_TOOL_NAME)
    async def _reset_extraction(_args: ResetExtractionArgs, _ctx: Any) -> ToolResult:
        """Rarely needed; use delete_node + upsert_node to repair instead. Drops all pending events / edges so you can re-emit the firing's graph from scratch. Reserved as a last-resort escape hatch when the accumulated draft is fundamentally unrecoverable — for ordinary fixes (merging neighbours, fixing a kind, repointing edges) the targeted edit tools are cheaper and preserve work."""
        if state.committed:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=(
                                "reset_extraction: firing already finalized; "
                                "reset is not possible after finalize_extraction succeeded"
                            ),
                            attempt="reset_extraction()",
                            state_echo=state_echo(state),
                            options=[
                                "stop calling tools — the firing has already terminated",
                            ],
                        ),
                    )
                ],
                is_error=True,
            )
        state.reset_pending()
        return ToolResult(content=[TextContent(type="text", text='{"ok": true, "reset": true}')])

    return _reset_extraction


__all__ = [
    "RESET_EXTRACTION_TOOL_NAME",
    "ResetExtractionArgs",
    "build_reset_extraction_tool",
]
