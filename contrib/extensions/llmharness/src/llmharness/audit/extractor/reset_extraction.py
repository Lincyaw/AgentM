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

from .._tool_decorator import harness_tool
from .._witness_errors import format_witness_error
from ._state_echo import state_echo
from .state import ExtractionState

RESET_EXTRACTION_TOOL_NAME = "reset_extraction"


class ResetExtractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_reset_extraction_tool(state: ExtractionState) -> FunctionTool:
    @harness_tool(RESET_EXTRACTION_TOOL_NAME)
    async def _reset_extraction(_args: ResetExtractionArgs, _ctx: Any) -> ToolResult:
        """Drop all pending events / edges so you can re-emit the firing's graph from scratch. Use this only when the accumulated graph is unrecoverable (e.g. you can't see a way to fix passthrough events without merging neighbours, which append-only can't do). The plan is preserved — only events / edges / drops are cleared."""
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
        return ToolResult(
            content=[TextContent(type="text", text='{"ok": true, "reset": true}')]
        )

    return _reset_extraction


__all__ = [
    "RESET_EXTRACTION_TOOL_NAME",
    "ResetExtractionArgs",
    "build_reset_extraction_tool",
]
