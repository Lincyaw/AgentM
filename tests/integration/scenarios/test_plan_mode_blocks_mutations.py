from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agentm.core.kernel import TextContent, ToolResultBlock, ToolResultMessage
from agentm.core.operations import BashOperations, ExecResult
from agentm.extensions.loader import load_scenario
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@dataclass
class RecordingBashOps(BashOperations):
    calls: list[tuple[str, str]]

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_data=None,
        signal=None,
    ) -> ExecResult:
        del timeout, env, on_data, signal
        self.calls.append((cmd, cwd))
        return ExecResult(stdout=b"ran", stderr=b"", exit_code=0, timed_out=False)


@pytest.mark.asyncio
async def test_plan_mode_blocks_mutating_bash_tool(tmp_path: Path) -> None:
    bash_ops = RecordingBashOps(calls=[])
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=load_scenario("plan_mode")
            + [
                (
                    "agentm.extensions.builtin.tool_bash",
                    {"bash_ops": bash_ops},
                )
            ],
            provider="scripted-fake",
            provider_config={
                "tool_name": "bash",
                "arguments": {"cmd": "echo should-not-run"},
                "final_text": "blocked",
            },
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        final = await session.prompt("try to mutate")
        tool_result_message = final[2]
        assert isinstance(tool_result_message, ToolResultMessage)
        result_block = tool_result_message.content[0]
        assert isinstance(result_block, ToolResultBlock)

        assert result_block.is_error is True
        assert isinstance(result_block.content[0], TextContent)
        assert "Tool call blocked" in result_block.content[0].text
        assert "denied by denylist" in result_block.content[0].text
        assert bash_ops.calls == []
    finally:
        await session.shutdown()
