"""Tests for SimpleAgentLoop — RED phase.

These tests define the expected behavior of the SimpleAgentLoop ReAct
implementation. They should all FAIL initially because the implementation
does not exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentm.harness.types import AgentEvent, AgentResult, AgentStatus, Message, RunConfig


# ---------------------------------------------------------------------------
# Mock helpers — lightweight fakes for LLM, tools, middleware
# ---------------------------------------------------------------------------


@dataclass
class MockToolCall:
    """Represents a single tool call in an AI response."""

    name: str
    args: dict[str, Any]
    id: str = "tc-1"


@dataclass
class MockAIResponse:
    """Minimal mock of an LLM response (AI message)."""

    content: str = "done"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    type: str = "ai"


class MockModel:
    """Mock ChatModel that returns pre-configured responses in order."""

    def __init__(self, responses: list[MockAIResponse]) -> None:
        self._responses = iter(responses)
        self.invocations: list[list[Any]] = []
        self._structured_calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> MockAIResponse:
        self.invocations.append(messages)
        return next(self._responses)

    def with_structured_output(self, schema: type, **kwargs: Any) -> MockStructuredModel:
        return MockStructuredModel(schema)


class MockStructuredModel:
    """Mock for model.with_structured_output(schema)."""

    def __init__(self, schema: type) -> None:
        self.schema = schema
        self.invocations: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.invocations.append(messages)
        return {"structured": True, "schema": self.schema.__name__}


class MockTool:
    """Mock tool with a fixed result."""

    def __init__(self, name: str, result: str = "tool result") -> None:
        self.name = name
        self._result = result
        self.invocations: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.invocations.append(args)
        return self._result


class PassthroughMiddleware:
    """Middleware that does nothing — passes everything through."""

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
        return await call_next(tool_name, tool_args)


class PrependMessageMiddleware:
    """Middleware that prepends a system message on llm_start."""

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


class ModifyResponseMiddleware:
    """Middleware that appends a suffix to response content on llm_end."""

    def __init__(self, suffix: str = " [modified]") -> None:
        self._suffix = suffix

    async def on_llm_start(self, messages: list[Any], ctx: Any) -> list[Any]:
        return messages

    async def on_llm_end(self, response: Any, ctx: Any) -> Any:
        response.content = response.content + self._suffix
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
    """Middleware that short-circuits tool calls with a cached result."""

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
        # Short-circuit: return cached result without calling call_next
        return self._cached


# ---------------------------------------------------------------------------
# Helper to build a SimpleAgentLoop
# ---------------------------------------------------------------------------

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
    """Create a SimpleAgentLoop with convenient defaults."""
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
    """Collect all events from stream()."""
    events = []
    async for event in loop.stream(input_val, config=config):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestSimpleAgentLoopBasicReact:
    """Test basic ReAct loop behavior."""

    @pytest.mark.asyncio
    async def test_basic_react_completes(self) -> None:
        """Model returns no tool_calls -> AgentResult with COMPLETED status."""
        model = MockModel([MockAIResponse(content="Final answer")])
        loop = _make_loop(model=model)

        result = await loop.run("What is 2+2?")

        assert result is not None
        assert result.status == AgentStatus.COMPLETED
        assert result.output == "Final answer"
        assert result.steps == 1
        assert result.tool_calls == 0

    @pytest.mark.asyncio
    async def test_tool_execution(self) -> None:
        """Model returns tool_call -> tool runs -> model called again -> completes."""
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
        assert len(tool.invocations) == 1
        assert tool.invocations[0] == {"expr": "2+2"}

    @pytest.mark.asyncio
    async def test_multi_step(self) -> None:
        """2 rounds of tool calls then final answer -> verify steps=3."""
        tool = MockTool("search", result="found it")
        model = MockModel([
            # Step 1: first tool call
            MockAIResponse(
                content="",
                tool_calls=[{"name": "search", "args": {"q": "first"}, "id": "tc-1"}],
            ),
            # Step 2: second tool call
            MockAIResponse(
                content="",
                tool_calls=[{"name": "search", "args": {"q": "second"}, "id": "tc-2"}],
            ),
            # Step 3: final answer
            MockAIResponse(content="All done"),
        ])
        loop = _make_loop(model=model, tools=[tool])

        result = await loop.run("Search twice")

        assert result.status == AgentStatus.COMPLETED
        assert result.steps == 3
        assert result.tool_calls == 2
        assert result.output == "All done"


class TestSimpleAgentLoopInbox:
    """Test inbox injection behavior."""

    @pytest.mark.asyncio
    async def test_inbox_injection(self) -> None:
        """inject() a message -> it appears in messages passed to LLM."""
        model = MockModel([
            # First call: has tool calls to allow a second LLM call
            MockAIResponse(
                content="",
                tool_calls=[{"name": "noop", "args": {}, "id": "tc-1"}],
            ),
            # Second call: completes
            MockAIResponse(content="done"),
        ])
        tool = MockTool("noop", result="ok")
        loop = _make_loop(model=model, tools=[tool])

        # Inject before running — message should be drained before the first LLM call
        loop.inject("urgent: check this")

        result = await loop.run("Do something")

        assert result.status == AgentStatus.COMPLETED
        # The injected message should have been in the messages list for the first LLM call
        first_call_messages = model.invocations[0]
        injected_found = any(
            "[Injected message]" in str(m) and "urgent: check this" in str(m)
            for m in first_call_messages
        )
        assert injected_found, "Injected message should appear in LLM messages"


class TestSimpleAgentLoopMiddleware:
    """Test middleware hook behavior."""

    @pytest.mark.asyncio
    async def test_middleware_llm_start(self) -> None:
        """Middleware prepends a message via on_llm_start."""
        model = MockModel([MockAIResponse(content="ok")])
        mw = PrependMessageMiddleware("[system] extra context")
        loop = _make_loop(model=model, middleware=[mw])

        await loop.run("Hello")

        # The first element in the messages passed to LLM should be the prepended one
        first_call_messages = model.invocations[0]
        assert first_call_messages[0] == "[system] extra context"

    @pytest.mark.asyncio
    async def test_middleware_llm_end(self) -> None:
        """Middleware modifies response content via on_llm_end."""
        model = MockModel([MockAIResponse(content="base response")])
        mw = ModifyResponseMiddleware(" [modified]")
        loop = _make_loop(model=model, middleware=[mw])

        result = await loop.run("Hello")

        assert result.output == "base response [modified]"

    @pytest.mark.asyncio
    async def test_middleware_tool_call_passthrough(self) -> None:
        """Middleware calls call_next normally — tool executes."""
        tool = MockTool("mytool", result="real result")
        model = MockModel([
            MockAIResponse(
                content="",
                tool_calls=[{"name": "mytool", "args": {}, "id": "tc-1"}],
            ),
            MockAIResponse(content="done"),
        ])
        mw = PassthroughMiddleware()
        loop = _make_loop(model=model, tools=[tool], middleware=[mw])

        result = await loop.run("Use the tool")

        assert result.status == AgentStatus.COMPLETED
        assert len(tool.invocations) == 1

    @pytest.mark.asyncio
    async def test_middleware_tool_call_shortcircuit(self) -> None:
        """Middleware returns cached result without calling call_next."""
        tool = MockTool("mytool", result="should not be called")
        model = MockModel([
            MockAIResponse(
                content="",
                tool_calls=[{"name": "mytool", "args": {}, "id": "tc-1"}],
            ),
            MockAIResponse(content="done with cached"),
        ])
        mw = CachingToolMiddleware("cached!")
        loop = _make_loop(model=model, tools=[tool], middleware=[mw])

        result = await loop.run("Use the tool")

        assert result.status == AgentStatus.COMPLETED
        # The actual tool should NOT have been called
        assert len(tool.invocations) == 0


class TestSimpleAgentLoopMaxSteps:
    """Test max_steps exhaustion."""

    @pytest.mark.asyncio
    async def test_max_steps_exhaustion(self) -> None:
        """max_steps=2, model always calls tools -> FAILED status."""
        tool = MockTool("loop_tool", result="again")
        # Model always returns tool calls — never gives a final answer
        model = MockModel([
            MockAIResponse(
                content="",
                tool_calls=[{"name": "loop_tool", "args": {}, "id": "tc-1"}],
            ),
            MockAIResponse(
                content="",
                tool_calls=[{"name": "loop_tool", "args": {}, "id": "tc-2"}],
            ),
            # This third response should never be reached
            MockAIResponse(content="unreachable"),
        ])
        loop = _make_loop(model=model, tools=[tool])
        config = RunConfig(max_steps=2)

        result = await loop.run("Keep looping", config=config)

        assert result.status == AgentStatus.FAILED
        assert result.error is not None
        assert "2" in result.error  # mentions the max_steps limit
        assert result.steps == 2


class TestSimpleAgentLoopOutputSchema:
    """Test structured output via output_schema."""

    @pytest.mark.asyncio
    async def test_output_schema(self) -> None:
        """output_schema set -> with_structured_output called at end."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str = ""

        model = MockModel([MockAIResponse(content="Final text")])
        loop = _make_loop(model=model, output_schema=MyOutput)

        result = await loop.run("Give structured output")

        assert result.status == AgentStatus.COMPLETED
        # The output should be the structured result, not raw text
        assert isinstance(result.output, dict)
        assert result.output.get("structured") is True
        assert result.output.get("schema") == "MyOutput"


class TestSimpleAgentLoopStreaming:
    """Test stream() yields correct event types."""

    @pytest.mark.asyncio
    async def test_stream_events(self) -> None:
        """stream() yields correct event types in order."""
        tool = MockTool("mytool", result="result")
        model = MockModel([
            MockAIResponse(
                content="",
                tool_calls=[{"name": "mytool", "args": {"x": 1}, "id": "tc-1"}],
            ),
            MockAIResponse(content="final"),
        ])
        loop = _make_loop(model=model, tools=[tool])

        events = await _collect_events(loop, "Do work")
        event_types = [e.type for e in events]

        # Expected sequence: llm_start, llm_end, tool_start, tool_end,
        #                    llm_start, llm_end, complete
        assert event_types == [
            "llm_start", "llm_end",
            "tool_start", "tool_end",
            "llm_start", "llm_end",
            "complete",
        ]

        # Verify the complete event carries the result
        complete_event = events[-1]
        assert complete_event.type == "complete"
        agent_result = complete_event.data.get("result")
        assert isinstance(agent_result, AgentResult)
        assert agent_result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_delegates_to_stream(self) -> None:
        """run() returns same result as collecting stream events."""
        model = MockModel([MockAIResponse(content="answer")])
        loop_a = _make_loop(model=model)

        # Use run()
        run_result = await loop_a.run("question")

        # Use stream() and extract result from complete event
        model_b = MockModel([MockAIResponse(content="answer")])
        loop_b = _make_loop(model=model_b)
        events = await _collect_events(loop_b, "question")
        complete_events = [e for e in events if e.type == "complete"]
        stream_result = complete_events[0].data.get("result")

        assert run_result.status == stream_result.status
        assert run_result.output == stream_result.output
        assert run_result.steps == stream_result.steps

    @pytest.mark.asyncio
    async def test_stream_inject_event(self) -> None:
        """stream() yields inject events when inbox has messages."""
        model = MockModel([
            MockAIResponse(
                content="",
                tool_calls=[{"name": "noop", "args": {}, "id": "tc-1"}],
            ),
            MockAIResponse(content="done"),
        ])
        tool = MockTool("noop", result="ok")
        loop = _make_loop(model=model, tools=[tool])
        loop.inject("external message")

        events = await _collect_events(loop, "Go")
        inject_events = [e for e in events if e.type == "inject"]

        assert len(inject_events) == 1
        assert inject_events[0].data.get("message") == "external message"


# ---------------------------------------------------------------------------
# Mock helpers for structured output retry/fallback tests
# ---------------------------------------------------------------------------


class MockFailThenSucceedStructuredModel:
    """Structured model that fails N times then succeeds."""

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
    """Structured model that always raises."""

    def __init__(self) -> None:
        self.invocations: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.invocations.append(messages)
        raise ValueError("Validation error: cannot parse output")


class MockModelWithStructuredControl(MockModel):
    """MockModel that returns a specific structured model from with_structured_output."""

    def __init__(
        self,
        responses: list[MockAIResponse],
        structured_model: Any,
    ) -> None:
        super().__init__(responses)
        self._structured_model = structured_model

    def with_structured_output(self, schema: type, **kwargs: Any) -> Any:
        return self._structured_model


# ---------------------------------------------------------------------------
# Tests for should_terminate
# ---------------------------------------------------------------------------


class TestShouldTerminate:
    """Test the should_terminate callback behavior."""

    @pytest.mark.asyncio
    async def test_should_terminate_skips_tool_calls(self) -> None:
        """When should_terminate returns True and response has tool_calls,
        tools are NOT executed and the loop finalizes."""
        tool = MockTool("calculator", result="4")
        model = MockModel([
            MockAIResponse(
                content="I want to finalize",
                tool_calls=[{"name": "calculator", "args": {"expr": "2+2"}, "id": "tc-1"}],
            ),
        ])
        # Always terminate regardless of tool_calls
        loop = _make_loop(
            model=model,
            tools=[tool],
            should_terminate=lambda resp: True,
        )

        result = await loop.run("Do math")

        assert result.status == AgentStatus.COMPLETED
        assert result.output == "I want to finalize"
        # Tool must NOT have been called
        assert len(tool.invocations) == 0
        assert result.tool_calls == 0

    @pytest.mark.asyncio
    async def test_should_not_terminate_continues_without_tools(self) -> None:
        """When should_terminate returns False and response has no tool_calls,
        the loop continues (calls LLM again) instead of finalizing."""
        call_count = 0

        def terminate_on_second(resp: Any) -> bool:
            nonlocal call_count
            call_count += 1
            # Don't terminate on first call, terminate on second
            return call_count >= 2

        model = MockModel([
            MockAIResponse(content="thinking..."),  # no tool_calls
            MockAIResponse(content="final answer"),  # no tool_calls
        ])
        loop = _make_loop(
            model=model,
            should_terminate=terminate_on_second,
        )

        result = await loop.run("Think hard")

        assert result.status == AgentStatus.COMPLETED
        assert result.output == "final answer"
        # LLM should have been called twice
        assert len(model.invocations) == 2
        assert result.steps == 2


# ---------------------------------------------------------------------------
# Tests for output_schema retry and fallback
# ---------------------------------------------------------------------------


class TestOutputSchemaRetryFallback:
    """Test structured output retry logic and raw_text fallback."""

    @pytest.mark.asyncio
    async def test_output_schema_retry_on_failure(self) -> None:
        """Structured output fails once, succeeds on retry."""
        from pydantic import BaseModel

        class Report(BaseModel):
            answer: str = ""

        success_result = {"answer": "the answer"}
        structured_model = MockFailThenSucceedStructuredModel(
            fail_count=1, success_result=success_result,
        )
        model = MockModelWithStructuredControl(
            responses=[MockAIResponse(content="done")],
            structured_model=structured_model,
        )
        loop = _make_loop(
            model=model,
            output_schema=Report,
            synthesize_retries=2,
        )

        result = await loop.run("Produce report")

        assert result.status == AgentStatus.COMPLETED
        assert result.output == {"answer": "the answer"}
        # Should have been called twice: first fail, then success
        assert len(structured_model.invocations) == 2

    @pytest.mark.asyncio
    async def test_output_schema_fallback_to_raw_text(self) -> None:
        """All structured output retries fail -> falls back to {"raw_text": ...}."""
        from pydantic import BaseModel

        class Report(BaseModel):
            answer: str = ""

        structured_model = MockAlwaysFailStructuredModel()
        # The fallback plain LLM call returns this
        fallback_response = MockAIResponse(content="fallback plain text")
        model = MockModelWithStructuredControl(
            responses=[
                MockAIResponse(content="initial response"),
                fallback_response,
            ],
            structured_model=structured_model,
        )
        loop = _make_loop(
            model=model,
            output_schema=Report,
            synthesize_retries=2,
        )

        result = await loop.run("Produce report")

        assert result.status == AgentStatus.COMPLETED
        assert isinstance(result.output, dict)
        assert "raw_text" in result.output
        assert result.output["raw_text"] == "fallback plain text"
        # Structured model: 1 initial + 2 retries = 3 calls
        assert len(structured_model.invocations) == 3


# ---------------------------------------------------------------------------
# Tests for list[Message] input path in SimpleAgentLoop
# ---------------------------------------------------------------------------


class TestSimpleAgentLoopListInput:
    """Test that stream()/run() accept list[Message] in addition to str.

    Bug prevented: Callers passing pre-built message lists (e.g. from
    AgentSystem._to_messages) would crash or lose messages if the loop
    only handled str input.
    """

    @pytest.mark.asyncio
    async def test_should_produce_same_result_for_str_and_single_message_list(self) -> None:
        """Passing [{"role": "human", "content": "hello"}] behaves
        identically to passing the string "hello"."""
        model_str = MockModel([MockAIResponse(content="answer-str")])
        loop_str = _make_loop(model=model_str, system_prompt="sys")

        model_list = MockModel([MockAIResponse(content="answer-list")])
        loop_list = _make_loop(model=model_list, system_prompt="sys")

        result_str = await loop_str.run("hello")
        result_list = await loop_list.run([{"role": "human", "content": "hello"}])

        # Both should complete successfully
        assert result_str.status == AgentStatus.COMPLETED
        assert result_list.status == AgentStatus.COMPLETED

        # Both should produce the same message structure sent to the LLM:
        # [system_prompt, human_message]
        msgs_str = model_str.invocations[0]
        msgs_list = model_list.invocations[0]
        assert len(msgs_str) == len(msgs_list)
        # First message is the system prompt in both cases
        assert msgs_str[0] == {"role": "system", "content": "sys"}
        assert msgs_list[0] == {"role": "system", "content": "sys"}
        # Second message is the human message
        assert msgs_str[1] == {"role": "human", "content": "hello"}
        assert msgs_list[1] == {"role": "human", "content": "hello"}

    @pytest.mark.asyncio
    async def test_should_preserve_all_messages_in_list_input(self) -> None:
        """Passing multiple messages preserves them all after the system prompt.

        Bug prevented: Only the first message is kept, or messages are
        flattened into a single string.
        """
        model = MockModel([MockAIResponse(content="done")])
        loop = _make_loop(model=model, system_prompt="You are helpful.")

        messages: list[Message] = [
            {"role": "human", "content": "Context: previous conversation"},
            {"role": "assistant", "content": "I understand"},
            {"role": "human", "content": "Now do the task"},
        ]
        result = await loop.run(messages)

        assert result.status == AgentStatus.COMPLETED
        # The first 4 messages sent to the LLM: system + all 3 user messages
        # (MockModel stores a reference, so the list grows after the call;
        #  we check the initial positions.)
        sent = model.invocations[0]
        assert sent[0] == {"role": "system", "content": "You are helpful."}
        assert sent[1] == {"role": "human", "content": "Context: previous conversation"}
        assert sent[2] == {"role": "assistant", "content": "I understand"}
        assert sent[3] == {"role": "human", "content": "Now do the task"}

    @pytest.mark.asyncio
    async def test_should_prepend_system_prompt_regardless_of_input_type(self) -> None:
        """System prompt is always the first message, whether input is
        str or list[Message].

        Bug prevented: System prompt omitted or duplicated when input
        is already a list containing a system message from the caller.
        """
        model = MockModel([MockAIResponse(content="ok")])
        loop = _make_loop(model=model, system_prompt="SYSTEM PROMPT")

        await loop.run([{"role": "human", "content": "task"}])

        sent = model.invocations[0]
        assert sent[0] == {"role": "system", "content": "SYSTEM PROMPT"}

    @pytest.mark.asyncio
    async def test_should_stream_events_correctly_with_list_input(self) -> None:
        """stream() with list input yields the same event sequence as str input.

        Bug prevented: Event generation breaks or changes when the input
        type changes from str to list.
        """
        tool = MockTool("mytool", result="result")
        model = MockModel([
            MockAIResponse(
                content="",
                tool_calls=[{"name": "mytool", "args": {"x": 1}, "id": "tc-1"}],
            ),
            MockAIResponse(content="final"),
        ])
        loop = _make_loop(model=model, tools=[tool])

        events = await _collect_events(
            loop, [{"role": "human", "content": "Do work"}]
        )
        event_types = [e.type for e in events]

        assert event_types == [
            "llm_start", "llm_end",
            "tool_start", "tool_end",
            "llm_start", "llm_end",
            "complete",
        ]

    @pytest.mark.asyncio
    async def test_should_handle_empty_list_input(self) -> None:
        """An empty message list still works -- system prompt is the only
        initial message sent to the LLM.

        Bug prevented: IndexError or empty-messages crash when caller
        passes [] (e.g. from _to_messages with no task).
        """
        model = MockModel([MockAIResponse(content="no input")])
        loop = _make_loop(model=model, system_prompt="sys")

        result = await loop.run([])

        assert result.status == AgentStatus.COMPLETED
        # The system prompt should be the first (and initially only) message
        sent = model.invocations[0]
        assert sent[0] == {"role": "system", "content": "sys"}
