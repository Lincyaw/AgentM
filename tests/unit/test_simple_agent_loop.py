"""Focused regression tests for `SimpleAgentLoop`."""
from __future__ import annotations

from typing import Any

import pytest

from agentm.harness.types import AgentEvent, AgentResult, AgentStatus, Message, RunConfig

from tests.helpers import MockAIResponse, MockModel, MockTool


class PrependMessageMiddleware:
    """Middleware that prepends a marker before the LLM call."""

    def __init__(self, message: str = "[middleware] injected") -> None:
        self._message = message

    async def on_llm_start(self, messages: list[Any], ctx: Any) -> list[Any]:
        return [self._message] + list(messages)

    async def on_llm_end(self, response: Any, ctx: Any) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Any,
        ctx: Any,
    ) -> str:
        return await call_next(tool_name, tool_args)


class CachingToolMiddleware:
    """Middleware that short-circuits tool execution."""

    def __init__(self, cached_result: str = "cached!") -> None:
        self._cached = cached_result

    async def on_llm_start(self, messages: list[Any], ctx: Any) -> list[Any]:
        return messages

    async def on_llm_end(self, response: Any, ctx: Any) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Any,
        ctx: Any,
    ) -> str:
        return self._cached


class CaptureToolContextMiddleware:
    """Records tool-call metadata seen by middleware."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def on_llm_start(self, messages: list[Any], ctx: Any) -> list[Any]:
        return messages

    async def on_llm_end(self, response: Any, ctx: Any) -> Any:
        return response

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Any,
        ctx: Any,
    ) -> str:
        self.calls.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_call_id": ctx.metadata.get("tool_call_id"),
            }
        )
        return await call_next(tool_name, tool_args)


class MockFailThenSucceedStructuredModel:
    """Structured model that fails N times before succeeding."""

    def __init__(self, fail_count: int, success_result: Any) -> None:
        self._fail_count = fail_count
        self._success_result = success_result
        self._call_count = 0
        self.invocations: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.invocations.append(messages)
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ValueError("Validation error: missing required field 'answer'")
        return self._success_result


class MockAlwaysFailStructuredModel:
    """Structured model that always fails validation."""

    def __init__(self) -> None:
        self.invocations: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.invocations.append(messages)
        raise ValueError("Validation error: cannot parse output")


class MockStructuredValidationError(Exception):
    """Validation error carrying the raw LLM output."""

    def __init__(self, message: str, llm_output: str) -> None:
        super().__init__(message)
        self.llm_output = llm_output


class MockAlwaysFailStructuredModelWithRaw:
    """Structured model that fails while preserving raw model output."""

    def __init__(self, llm_output: str) -> None:
        self.invocations: list[list[Any]] = []
        self._llm_output = llm_output

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.invocations.append(messages)
        raise MockStructuredValidationError(
            "Validation error: cannot parse output",
            llm_output=self._llm_output,
        )


class MockModelWithStructuredControl(MockModel):
    """Mock model that returns a caller-supplied structured-output wrapper."""

    def __init__(self, responses: list[MockAIResponse], structured_model: Any) -> None:
        super().__init__(responses)
        self._structured_model = structured_model

    def with_structured_output(self, schema: type, **kwargs: Any) -> Any:
        return self._structured_model


def _make_loop(
    *,
    model: MockModel | None = None,
    tools: list[MockTool] | None = None,
    system_prompt: str = "You are a test agent.",
    middleware: list[Any] | None = None,
    output_schema: type | None = None,
    output_prompt: str = "",
    synthesize_retries: int = 2,
    should_terminate: Any = None,
) -> Any:
    from agentm.harness.loops.simple import SimpleAgentLoop

    return SimpleAgentLoop(
        model=model or MockModel([MockAIResponse()]),
        tools=tools or [],
        system_prompt=system_prompt,
        middleware=middleware,
        output_schema=output_schema,
        output_prompt=output_prompt,
        synthesize_retries=synthesize_retries,
        should_terminate=should_terminate,
    )


async def _collect_events(
    loop: Any,
    input_val: str | list[Message],
    config: RunConfig | None = None,
) -> list[AgentEvent]:
    events = []
    async for event in loop.stream(input_val, config=config):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_completes_without_tool_calls() -> None:
    loop = _make_loop(model=MockModel([MockAIResponse(content="Final answer")]))

    result = await loop.run("What is 2+2?")

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "Final answer"
    assert result.steps == 1
    assert result.tool_calls == 0


@pytest.mark.asyncio
async def test_executes_tool_call_and_returns_second_llm_answer() -> None:
    tool = MockTool("calculator", result="4")
    model = MockModel([
        MockAIResponse(
            content="",
            tool_calls=[{"name": "calculator", "args": {"expr": "2+2"}, "id": "tc-1"}],
        ),
        MockAIResponse(content="The answer is 4"),
    ])
    loop = _make_loop(model=model, tools=[tool])

    result = await loop.run("What is 2+2?")

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "The answer is 4"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert tool.invocations == [{"expr": "2+2"}]


@pytest.mark.asyncio
async def test_injected_message_is_delivered_before_llm_call() -> None:
    model = MockModel([
        MockAIResponse(content="", tool_calls=[{"name": "noop", "args": {}, "id": "tc-1"}]),
        MockAIResponse(content="done"),
    ])
    loop = _make_loop(model=model, tools=[MockTool("noop", result="ok")])
    loop.inject("urgent: check this")

    result = await loop.run("Do something")

    assert result.status == AgentStatus.COMPLETED
    first_call_messages = model.invocations[0]
    assert any(
        "[Injected message]" in str(message) and "urgent: check this" in str(message)
        for message in first_call_messages
    )


@pytest.mark.asyncio
async def test_middleware_can_augment_prompt_before_llm_call() -> None:
    model = MockModel([MockAIResponse(content="ok")])
    loop = _make_loop(model=model, middleware=[PrependMessageMiddleware("[system] extra")])

    await loop.run("Hello")

    assert model.invocations[0][0] == "[system] extra"


@pytest.mark.asyncio
async def test_middleware_can_short_circuit_tool_execution() -> None:
    tool = MockTool("mytool", result="should not be called")
    model = MockModel([
        MockAIResponse(content="", tool_calls=[{"name": "mytool", "args": {}, "id": "tc-1"}]),
        MockAIResponse(content="done with cached"),
    ])
    loop = _make_loop(model=model, tools=[tool], middleware=[CachingToolMiddleware("cached!")])

    result = await loop.run("Use the tool")

    assert result.status == AgentStatus.COMPLETED
    assert tool.invocations == []


@pytest.mark.asyncio
async def test_stream_emits_expected_event_sequence_for_single_tool_turn() -> None:
    tool = MockTool("mytool", result="result")
    model = MockModel([
        MockAIResponse(content="", tool_calls=[{"name": "mytool", "args": {"x": 1}, "id": "tc-1"}]),
        MockAIResponse(content="final"),
    ])
    loop = _make_loop(model=model, tools=[tool])

    events = await _collect_events(loop, "Do work")

    assert [event.type for event in events] == [
        "llm_start",
        "llm_end",
        "tool_start",
        "tool_end",
        "llm_start",
        "llm_end",
        "complete",
    ]
    assert isinstance(events[-1].data.get("result"), AgentResult)


@pytest.mark.asyncio
async def test_tool_call_id_is_available_in_tool_middleware_context() -> None:
    tool = MockTool("mytool", result="result")
    capture = CaptureToolContextMiddleware()
    model = MockModel([
        MockAIResponse(
            content="",
            tool_calls=[{"name": "mytool", "args": {"x": 1}, "id": "tc-1"}],
        ),
        MockAIResponse(content="final"),
    ])
    loop = _make_loop(model=model, tools=[tool], middleware=[capture])

    result = await loop.run("Do work")

    assert result.status == AgentStatus.COMPLETED
    assert capture.calls == [
        {
            "tool_name": "mytool",
            "tool_args": {"x": 1},
            "tool_call_id": "tc-1",
        }
    ]


@pytest.mark.asyncio
async def test_empty_final_response_logs_diagnostic_and_returns_failed_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = MockModel([
        MockAIResponse(
            content="",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "debug-model",
                "id": "resp-empty-1",
            },
        )
    ])
    loop = _make_loop(model=model)

    with caplog.at_level("ERROR"):
        result = await loop.run("Do work")

    assert result.status == AgentStatus.FAILED
    assert result.error == "Synthesis produced empty non-schema output"
    assert "empty llm response before termination" in caplog.text
    assert "debug-model" in caplog.text
    assert "resp-empty-1" in caplog.text


@pytest.mark.asyncio
async def test_trajectory_middleware_records_empty_llm_response(tmp_path) -> None:
    from agentm.core.trajectory import TrajectoryCollector
    from agentm.harness.middleware import TrajectoryMiddleware

    trajectory = TrajectoryCollector(run_id="run-empty", output_dir=str(tmp_path))
    model = MockModel([
        MockAIResponse(
            content="",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "debug-model",
                "id": "resp-empty-2",
            },
        )
    ])
    loop = _make_loop(
        model=model,
        middleware=[TrajectoryMiddleware(trajectory, agent_path=["orchestrator"])],
    )

    result = await loop.run("Do work")

    assert result.status == AgentStatus.FAILED
    empty_event = next(
        event for event in trajectory.events if event.get("event_type") == "llm_end_empty"
    )
    assert empty_event["agent_path"] == ["orchestrator"]
    assert empty_event["data"]["finish_reason"] == "stop"
    assert empty_event["data"]["model_name"] == "debug-model"
    assert empty_event["data"]["response_id"] == "resp-empty-2"


@pytest.mark.asyncio
async def test_output_schema_failures_produce_failed_result_instead_of_plain_fallback() -> None:
    from pydantic import BaseModel

    class Report(BaseModel):
        answer: str = ""

    model = MockModelWithStructuredControl(
        responses=[MockAIResponse(content="initial response")],
        structured_model=MockAlwaysFailStructuredModel(),
    )
    loop = _make_loop(model=model, output_schema=Report, synthesize_retries=2)

    result = await loop.run("Produce report")

    assert result.status == AgentStatus.FAILED
    assert result.output is None
    assert "Synthesis failed after 3 attempts" in (result.error or "")


@pytest.mark.asyncio
async def test_output_schema_uses_last_raw_output_in_failure_message() -> None:
    from pydantic import BaseModel

    class Report(BaseModel):
        answer: str = ""

    raw_graph = '{"nodes":[],"edges":[],"root_causes":[],"component_to_service":{}}'
    model = MockModelWithStructuredControl(
        responses=[MockAIResponse(content="initial response")],
        structured_model=MockAlwaysFailStructuredModelWithRaw(llm_output=raw_graph),
    )
    loop = _make_loop(model=model, output_schema=Report, synthesize_retries=2)

    result = await loop.run("Produce report")

    assert result.status == AgentStatus.FAILED
    assert result.output is None
    assert "payload_preview" not in (result.error or "")
    assert "Synthesis failed after 3 attempts" in (result.error or "")


@pytest.mark.asyncio
async def test_max_steps_exhaustion_returns_failed_result() -> None:
    tool = MockTool("loop_tool", result="again")
    model = MockModel([
        MockAIResponse(content="", tool_calls=[{"name": "loop_tool", "args": {}, "id": "tc-1"}]),
        MockAIResponse(content="", tool_calls=[{"name": "loop_tool", "args": {}, "id": "tc-2"}]),
        MockAIResponse(content="unreachable"),
    ])
    loop = _make_loop(model=model, tools=[tool])

    result = await loop.run("Keep looping", config=RunConfig(max_steps=2))

    assert result.status == AgentStatus.FAILED
    assert result.error == "Max steps (2) reached"
    assert result.steps == 2


@pytest.mark.asyncio
async def test_should_terminate_can_finalize_without_running_tools() -> None:
    tool = MockTool("calculator", result="4")
    model = MockModel([
        MockAIResponse(
            content="I want to finalize",
            tool_calls=[{"name": "calculator", "args": {"expr": "2+2"}, "id": "tc-1"}],
        )
    ])
    loop = _make_loop(model=model, tools=[tool], should_terminate=lambda resp: True)

    result = await loop.run("Do math")

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "I want to finalize"
    assert result.tool_calls == 0
    assert tool.invocations == []


@pytest.mark.asyncio
async def test_should_terminate_false_with_no_tool_calls_continues_loop() -> None:
    call_count = 0

    def terminate_on_second(_: Any) -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 2

    model = MockModel([
        MockAIResponse(content="thinking..."),
        MockAIResponse(content="final answer"),
    ])
    loop = _make_loop(model=model, should_terminate=terminate_on_second)

    result = await loop.run("Think hard")

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "final answer"
    assert len(model.invocations) == 2
    assert result.steps == 2


@pytest.mark.asyncio
async def test_output_schema_retries_before_returning_structured_result() -> None:
    from pydantic import BaseModel

    class Report(BaseModel):
        answer: str = ""

    structured_model = MockFailThenSucceedStructuredModel(
        fail_count=1,
        success_result={"answer": "the answer"},
    )
    model = MockModelWithStructuredControl(
        responses=[MockAIResponse(content="done")],
        structured_model=structured_model,
    )
    loop = _make_loop(model=model, output_schema=Report, synthesize_retries=2)

    result = await loop.run("Produce report")

    assert result.status == AgentStatus.COMPLETED
    assert result.output == {"answer": "the answer"}
    assert len(structured_model.invocations) == 2


@pytest.mark.asyncio
async def test_list_input_strips_caller_system_messages_and_preserves_history() -> None:
    model = MockModel([MockAIResponse(content="done")])
    loop = _make_loop(model=model, system_prompt="Owned by loop")

    await loop.run([
        {"role": "system", "content": "caller system should be ignored"},
        {"role": "human", "content": "Context: previous conversation"},
        {"role": "assistant", "content": "I understand"},
        {"role": "human", "content": "Now do the task"},
    ])

    assert model.invocations[0][:4] == [
        {"role": "system", "content": "Owned by loop"},
        {"role": "human", "content": "Context: previous conversation"},
        {"role": "assistant", "content": "I understand"},
        {"role": "human", "content": "Now do the task"},
    ]
