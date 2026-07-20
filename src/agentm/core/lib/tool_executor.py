"""Backend-neutral direct tool execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    ToolExecutionCapabilities,
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
        return ToolExecutionCapabilities(
            isolation=("none",),
            filesystem=("none", "read", "write"),
            network=True,
            interrupt=("block", "cancel"),
        )

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


def _validate_requirements(
    requirements: ToolExecutionRequirements,
    capabilities: ToolExecutionCapabilities,
) -> None:
    unsupported: list[str] = []
    if requirements.isolation not in capabilities.isolation:
        unsupported.append(f"isolation={requirements.isolation}")
    if requirements.filesystem not in capabilities.filesystem:
        unsupported.append(f"filesystem={requirements.filesystem}")
    if requirements.killable and not capabilities.killable:
        unsupported.append("killable=true")
    if requirements.network and not capabilities.network:
        unsupported.append("network=true")
    if requirements.concurrency not in capabilities.concurrency:
        unsupported.append(f"concurrency={requirements.concurrency}")
    if requirements.interrupt not in capabilities.interrupt:
        unsupported.append(f"interrupt={requirements.interrupt}")
    if requirements.environment_id is not None and (
        capabilities.environment is None
        or capabilities.environment.id != requirements.environment_id
    ):
        unsupported.append(f"environment_id={requirements.environment_id}")
    if unsupported:
        raise RuntimeError(
            "tool executor does not satisfy requirements: " + ", ".join(unsupported)
        )


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
    capabilities = chosen.capabilities()
    _validate_requirements(resolved_requirements, capabilities)
    request = ToolExecutionRequest(
        tool=tool,
        args=args,
        requirements=resolved_requirements,
        environment=capabilities.environment,
    )
    return await chosen.execute(request, signal=signal)


__all__ = [
    "DirectToolExecutor",
    "execute_tool_call",
]
