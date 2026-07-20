# code-health: ignore-file[AM025] -- execution worker and wire boundaries validate cross-process payloads
"""Environment-backed sandbox ``ToolExecutor``."""

from __future__ import annotations

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.tool import ToolOutcome, ToolResult
from agentm.core.abi.tool_executor import (
    EnvironmentExecutableTool,
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
        if isinstance(request.tool, EnvironmentExecutableTool):
            return await request.tool.execute_in_environment(
                request.args,
                environment=self._environment,
                cwd=request.cwd,
                signal=signal,
            )
        if requirements.isolation == "environment":
            raise RuntimeError(
                f"tool {request.tool.name!r} requires environment isolation "
                "but does not implement EnvironmentExecutableTool"
            )
        return await self._direct.execute(request, signal=signal)


__all__ = ["SandboxToolExecutor"]
