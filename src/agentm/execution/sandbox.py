"""Environment-backed sandbox ``ToolExecutor``."""

from __future__ import annotations

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import TextContent
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.tool import ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    ToolExecutionCapabilities,
    ToolExecutionRequest,
)
from agentm.core.lib.tool_executor import DirectToolExecutor


class SandboxToolExecutor:
    """Execute environment-aware tools through ``EnvironmentOperations``."""

    def __init__(self, environment: EnvironmentOperations) -> None:
        self._environment = environment
        self._direct = DirectToolExecutor()

    def capabilities(self) -> ToolExecutionCapabilities:
        return ToolExecutionCapabilities(
            environment=self._environment.ref,
            isolation=("none", "environment"),
            filesystem=("none", "read", "write"),
            killable=True,
            network=True,
            concurrency=("exclusive", "parallel_safe"),
            interrupt=("block", "cancel"),
        )

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        requirements = request.requirements
        if requirements is not None and requirements.isolation == "environment":
            return await self._execute_environment_tool(request, signal=signal)
        if request.tool.name == "bash" and "cmd" in request.args:
            return await self._execute_bash(request, signal=signal)
        return await self._direct.execute(request, signal=signal)

    async def _execute_environment_tool(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None,
    ) -> ToolResult | ToolOutcome:
        if request.tool.name == "bash" and "cmd" in request.args:
            return await self._execute_bash(request, signal=signal)
        return await self._direct.execute(request, signal=signal)

    async def _execute_bash(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None,
    ) -> ToolResult:
        cmd = request.args.get("cmd")
        if not isinstance(cmd, str):
            return ToolResult(
                content=[TextContent(type="text", text="bash cmd must be a string")],
                is_error=True,
            )
        timeout = request.args.get("timeout")
        cwd = request.cwd or str(self._environment.ref.metadata.get("cwd", ""))
        result = await self._environment.bash.exec(
            cmd,
            cwd=cwd,
            timeout=timeout if isinstance(timeout, (int, float)) else None,
            signal=signal,
        )
        text = result.stdout.decode("utf-8", errors="replace")
        if result.stderr:
            text = f"{text}\n{result.stderr.decode('utf-8', errors='replace')}"
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=result.exit_code != 0 or result.timed_out,
            extras={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "environment_id": self._environment.ref.id,
            },
        )


__all__ = ["SandboxToolExecutor"]
