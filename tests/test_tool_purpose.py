"""Behavior contracts for the builtin tool_purpose atom."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agentm.core.abi import BeforeSendEvent, CancelSignal, FunctionTool, TextContent
from agentm.core.abi import ToolExecutionCapabilities, ToolExecutionRequest
from agentm.core.abi import ToolExecutor, ToolResult
from agentm.core.abi.tool_executor import ToolExecutionRequirements
from agentm.extensions.builtin.tool_purpose import ToolPurposeConfig, install


class _FakeAPI:
    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self.handlers: dict[str, Callable[[BeforeSendEvent], object]] = {}
        self._executor = executor
        self.registered_executor: ToolExecutor | None = None

    def on(
        self,
        channel: str,
        handler: Callable[[BeforeSendEvent], object],
        *,
        priority: int = 500,
    ) -> None:
        del priority
        self.handlers[channel] = handler

    def get_tool_executor(self) -> ToolExecutor | None:
        return self._executor

    def register_tool_executor(
        self, executor: ToolExecutor, *, replace: bool = False
    ) -> None:
        assert replace
        self.registered_executor = executor


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def capabilities(self) -> ToolExecutionCapabilities:
        return ToolExecutionCapabilities()

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult:
        del signal
        self.calls.append(dict(request.args))
        return ToolResult([TextContent(type="text", text="ok")])


async def _ok(args: dict[str, object]) -> ToolResult:
    del args
    return ToolResult([TextContent(type="text", text="ok")])


def _tool(
    *,
    name: str = "read",
    parameters: dict[str, object] | None = None,
) -> FunctionTool:
    return FunctionTool(
        name=name,
        description="Test tool",
        parameters=parameters
        or {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        fn=_ok,
        metadata={"file_op": "read"},
        execution_requirements=ToolExecutionRequirements(filesystem="read"),
    )


def _install_api(
    config: ToolPurposeConfig,
    *,
    executor: _RecordingExecutor | None = None,
) -> _FakeAPI:
    api = _FakeAPI(executor)
    install(api, config)  # type: ignore[arg-type]
    return api


def test_tool_purpose_injects_required_purpose_in_place() -> None:
    tool = _tool()
    api = _install_api(ToolPurposeConfig())
    handler = api.handlers[BeforeSendEvent.CHANNEL]

    result = handler(BeforeSendEvent(tools=(tool,)))

    assert result is None
    properties = tool.parameters["properties"]
    assert isinstance(properties, dict)
    assert "purpose" in properties
    assert tool.parameters["required"] == ["path", "purpose"]


def test_tool_purpose_mutation_composes_with_precomputed_tool_filter_list() -> None:
    tool = _tool()
    filtered_tools = [tool]
    api = _install_api(ToolPurposeConfig())
    handler = api.handlers[BeforeSendEvent.CHANNEL]

    result = handler(BeforeSendEvent(tools=(tool,)))

    assert result is None
    properties = filtered_tools[0].parameters["properties"]
    assert isinstance(properties, dict)
    assert "purpose" in properties


@pytest.mark.asyncio
async def test_tool_purpose_executor_strips_synthetic_arg() -> None:
    executor = _RecordingExecutor()
    tool = _tool()
    api = _install_api(ToolPurposeConfig(), executor=executor)
    handler = api.handlers[BeforeSendEvent.CHANNEL]
    handler(BeforeSendEvent(tools=(tool,)))

    assert api.registered_executor is not None
    await api.registered_executor.execute(
        ToolExecutionRequest(
            tool=tool,
            args={"path": "note.txt", "purpose": "inspect note"},
        )
    )

    assert executor.calls == [{"path": "note.txt"}]


def test_tool_purpose_honors_exclude() -> None:
    tool = _tool(name="finish")
    api = _install_api(ToolPurposeConfig(exclude=["finish"]))
    handler = api.handlers[BeforeSendEvent.CHANNEL]

    result = handler(BeforeSendEvent(tools=(tool,)))

    assert result is None
    properties = tool.parameters["properties"]
    assert isinstance(properties, dict)
    assert "purpose" not in properties


def test_tool_purpose_preserves_native_purpose_parameter() -> None:
    tool = _tool(
        parameters={
            "type": "object",
            "properties": {
                "purpose": {"type": "string"},
            },
            "required": ["purpose"],
        }
    )
    api = _install_api(ToolPurposeConfig())
    handler = api.handlers[BeforeSendEvent.CHANNEL]

    result = handler(BeforeSendEvent(tools=(tool,)))

    assert result is None
    assert tool.parameters["required"] == ["purpose"]
