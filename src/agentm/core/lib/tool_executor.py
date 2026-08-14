"""Backend-neutral direct tool execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import JsonValue, thaw_json
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    ToolExecutionRequest,
    ToolExecutionRequirements,
    ToolExecutor,
    ToolInterruptBehavior,
    tool_execution_requirements,
)


async def _execute_direct(
    tool: Tool,
    args: Mapping[str, object],
    *,
    signal: CancelSignal | None,
    interrupt: ToolInterruptBehavior,
) -> ToolResult | ToolOutcome:
    # Requests deep-freeze their args for audit immutability; tools are
    # promised plain JSON containers, so thaw at the delivery boundary —
    # a defensive deep copy that keeps the frozen original on the request.
    frozen_args = cast("JsonValue", args)
    thawed_args = cast("dict[str, object]", thaw_json(frozen_args))
    task = asyncio.create_task(
        tool.execute(thawed_args, signal=signal),
        name=f"agentm-tool-{tool.name}",
    )
    signal_task: asyncio.Task[object] | None = None
    try:
        if signal is not None and interrupt == "cancel":
            signal_task = asyncio.create_task(
                signal.wait(),
                name=f"agentm-tool-signal-{tool.name}",
            )
            done, _ = await asyncio.wait(
                {task, signal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if signal_task in done and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise asyncio.CancelledError("tool interrupted")
        return await task
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        if signal_task is not None and not signal_task.done():
            signal_task.cancel()
            await asyncio.gather(signal_task, return_exceptions=True)


class DirectToolExecutor:
    """Default executor: run the tool coroutine in the current event loop."""

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        return await _execute_direct(
            request.tool,
            request.args,
            signal=signal,
            interrupt=request.requirements.interrupt,
        )


_DIRECT_EXECUTOR = DirectToolExecutor()


async def execute_tool_call(
    tool: Tool,
    args: Mapping[str, object],
    *,
    signal: CancelSignal | None,
    executor: ToolExecutor | None = None,
    requirements: ToolExecutionRequirements | None = None,
) -> ToolResult | ToolOutcome:
    """Execute one tool call through the configured executor boundary."""

    resolved_requirements = (
        requirements if requirements is not None else tool_execution_requirements(tool)
    )
    chosen = _DIRECT_EXECUTOR if executor is None else executor
    request = ToolExecutionRequest(
        tool=tool,
        args=args,
        requirements=resolved_requirements,
    )
    return await chosen.execute(request, signal=signal)


__all__ = [
    "DirectToolExecutor",
    "execute_tool_call",
]
