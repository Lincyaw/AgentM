from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from agentm.core.kernel import AgentStartEvent, FunctionTool, TextContent, ToolResult
from agentm.harness.extension import ExtensionLoadError
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


@pytest.mark.asyncio
async def test_file_mutation_queue_serializes_same_path_mutations(tmp_path: Path) -> None:
    module_name = "tests.unit.extensions.builtin.file_mutation_queue._tools_ext"
    module = types.ModuleType(module_name)
    events: list[str] = []
    in_flight = 0
    overlap_detected = False

    async def make_result(args: dict[str, object], tool_name: str) -> ToolResult:
        nonlocal in_flight, overlap_detected
        if in_flight:
            overlap_detected = True
        in_flight += 1
        events.append(f"start:{tool_name}")
        await asyncio.sleep(0.01)
        events.append(f"end:{tool_name}")
        in_flight -= 1
        return ToolResult(content=[TextContent(type="text", text=str(args["path"]))])

    def install(api: object, config: dict[str, object]) -> None:
        api.register_tool(  # type: ignore[attr-defined]
            FunctionTool(
                name="edit",
                description="edit",
                parameters={"type": "object"},
                fn=lambda args: make_result(args, "edit"),
            )
        )
        api.register_tool(  # type: ignore[attr-defined]
            FunctionTool(
                name="write",
                description="write",
                parameters={"type": "object"},
                fn=lambda args: make_result(args, "write"),
            )
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = module

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (module_name, {}),
                ("agentm.extensions.builtin.file_mutation_queue", {}),
            ],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    returns = await session.bus.emit("agent_start", AgentStartEvent(messages=[]))
    assert returns == [None]

    tools = {tool.name: tool for tool in session.tools}
    await asyncio.gather(
        tools["edit"].execute({"path": "/tmp/x"}),
        tools["write"].execute({"path": "/tmp/x"}),
    )

    assert not overlap_detected
    assert events in (["start:edit", "end:edit", "start:write", "end:write"], ["start:write", "end:write", "start:edit", "end:edit"])
    await session.shutdown()


@pytest.mark.asyncio
async def test_file_mutation_queue_fast_fails_when_loaded_before_tools(tmp_path: Path) -> None:
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.file_mutation_queue", {}),
            ],
            provider="recording",
            resource_loader=InMemoryResourceLoader(),
        )
    )

    with pytest.raises(ExtensionLoadError):
        await session.prompt("hello")

    await session.shutdown()
