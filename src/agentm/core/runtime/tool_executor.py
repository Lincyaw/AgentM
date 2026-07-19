"""Runtime-owned tool execution boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    ToolExecutionCapabilities,
    ToolExecutionRequest,
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
    task = asyncio.create_task(
        tool.execute(dict(args), signal=signal),
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

    def capabilities(self) -> ToolExecutionCapabilities:
        return ToolExecutionCapabilities(interrupt=("block", "cancel"))

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        interrupt = request.requirements.interrupt if request.requirements else "block"
        return await _execute_direct(
            request.tool,
            request.args,
            signal=signal,
            interrupt=interrupt,
        )


_DIRECT_EXECUTOR = DirectToolExecutor()


async def execute_tool_call(
    tool: Tool,
    args: Mapping[str, object],
    *,
    signal: CancelSignal | None,
    executor: ToolExecutor | None = None,
) -> ToolResult | ToolOutcome:
    """Execute one tool call through the configured executor boundary."""

    request = ToolExecutionRequest(
        tool=tool,
        args=args,
        requirements=tool_execution_requirements(tool),
    )
    chosen = executor or _DIRECT_EXECUTOR
    return await chosen.execute(request, signal=signal)


__all__ = [
    "DirectToolExecutor",
    "execute_tool_call",
]
