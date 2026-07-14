"""Explicit ``finish`` tool with a submit confirmation.

The reference agent (terminus_2) ends a task with a deliberate
``task_complete`` signal and is asked to re-assert it once before its work is
graded. Our native loop has no such deliberate signal -- it just stops
emitting tool calls -- so there is nothing clean to confirm, and hooking the
"stopped calling tools" moment second-guesses every natural pause.

This atom instead registers a ``finish`` tool the agent calls to submit. The
first call returns a confirmation prompt and keeps the session alive; only a
repeat call ends the loop (via :class:`ToolTerminate`). That gives the same
"are you sure?" safety net on a deliberate action. An agent that simply stops
without calling ``finish`` ends as before -- the tool only adds a confirmed
submit path, it does not force one.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

_CONFIRM_TEXT = (
    "Are you sure the task is complete? Calling finish again submits your "
    "work for grading and cannot be undone. If anything the task asked for is "
    "unfinished, untested, or unverified, keep working and check it. If you "
    "are certain everything is done and verified, call finish again to submit."
)


class CompletionGuardConfig(BaseModel):
    # Number of confirmation prompts before a finish call ends the loop.
    confirmations: int = 1


class _FinishParams(BaseModel):
    """The finish tool takes no arguments."""


MANIFEST = ExtensionManifest(
    name="completion_guard",
    description=(
        "Register a `finish` tool the agent calls to submit its work; the "
        "first call asks for confirmation and keeps the session alive, a "
        "repeat call ends the loop (mirrors terminus_2's submit double-confirm)."
    ),
    registers=("tool:finish",),
    config_schema=CompletionGuardConfig,
    requires=(),
)

class _FinishRuntime:
    def __init__(self, api: ExtensionAPI, config: CompletionGuardConfig) -> None:
        self._api = api
        self._confirmations = max(0, config.confirmations)
        self._calls = 0

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="finish",
                description=(
                    "Call this when you believe the task is fully complete and "
                    "verified. It submits your work for grading. You are asked "
                    "to confirm once; call finish again to submit."
                ),
                parameters=pydantic_to_tool_schema(_FinishParams),
                fn=self._finish,
            )
        )

    async def _finish(self, args: dict[str, object]) -> ToolResult | ToolTerminate:
        self._calls += 1
        if self._calls <= self._confirmations:
            logger.info(
                "completion_guard: finish called ({}), asking to confirm",
                self._calls,
            )
            return ToolResult(
                content=[TextContent(type="text", text=_CONFIRM_TEXT)],
                is_error=False,
            )
        logger.info("completion_guard: finish confirmed; terminating loop")
        return ToolTerminate(
            result=ToolResult(
                content=[
                    TextContent(type="text", text="Task submitted for grading.")
                ],
                is_error=False,
            ),
            reason="terminal_bench:finish-confirmed",
        )


def install(api: ExtensionAPI, config: CompletionGuardConfig) -> None:
    _FinishRuntime(api, config).install()
