"""Atom: slow_tool — a tool that sleeps until cancelled.

Registers a ``slow_compute`` tool that blocks for a configurable duration,
cooperatively checking the cancellation signal.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import AtomAPI
from agentm.core.abi.cancel import CancelSignal, cancel_reason
from agentm.core.abi.manifest import AtomInstallPriority, ExtensionManifest
from agentm.core.abi.messages import TextContent
from agentm.core.abi.tool import ToolResult
from agentm.core.lib.tool_schema import pydantic_to_tool_schema


class SlowToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_duration: float = 30.0


MANIFEST = ExtensionManifest(
    name="slow_tool",
    description="Register a slow_compute tool that respects cancellation signals.",
    registers=("tool:slow_compute",),
    config_schema=SlowToolConfig,
    requires=(),
    priority=AtomInstallPriority.TOOL,
)


class SlowComputeArgs(BaseModel):
    seconds: float = Field(
        default=30.0,
        description="How many seconds to compute (blocks until done or cancelled).",
    )


class _SlowComputeTool:
    name = "slow_compute"
    description = (
        "Simulate a long-running computation. "
        "Blocks for the given duration or until interrupted."
    )
    parameters = pydantic_to_tool_schema(SlowComputeArgs)

    def __init__(self, default_duration: float) -> None:
        self._default_duration = default_duration

    async def execute(
        self,
        args: dict[str, object],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult:
        parsed = SlowComputeArgs.model_validate(args)
        duration = parsed.seconds or self._default_duration

        if signal is not None:
            try:
                await asyncio.wait_for(signal.wait(), timeout=duration)
            except TimeoutError:
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"computation completed after {duration}s",
                    )],
                )
            reason = cancel_reason(signal) or "unknown"
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"computation cancelled (reason: {reason})",
                )],
                is_error=True,
            )

        await asyncio.sleep(duration)
        return ToolResult(
            content=[TextContent(
                type="text",
                text=f"computation completed after {duration}s",
            )],
        )


def install(session: AtomAPI, config: SlowToolConfig) -> None:
    session.register_tool(_SlowComputeTool(config.default_duration))
