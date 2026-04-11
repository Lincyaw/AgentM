"""Shared test helpers for AgentM test suite.

Provides unified fake/mock classes used across multiple test files:
- FakeAgentLoop: configurable agent loop for testing
- NeverEndingLoop: loop that hangs until cancelled
- MockAIResponse: minimal mock of an LLM response
- MockModel: mock ChatModel with pre-configured response sequence
- MockTool: mock tool with a fixed result
- MockEventHandler: collects all events for assertion
- FakeWorkerFactory: mock factory that produces FakeAgentLoop instances
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agentm.harness.types import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    Message,
    RunConfig,
)


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------


@dataclass
class MockAIResponse:
    """Minimal mock of an LLM AI response."""

    content: str = "done"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    type: str = "ai"
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    content_blocks: list[Any] = field(default_factory=list)
    text: str = ""


class MockModel:
    """Mock ChatModel that returns pre-configured responses in sequence.

    Supports both required-list and optional-list constructors:
    - MockModel([resp1, resp2])          -- explicit sequence
    - MockModel()                        -- single default response
    """

    def __init__(self, responses: list[MockAIResponse] | None = None) -> None:
        self._responses = iter(responses or [MockAIResponse()])
        self.invocations: list[list[Any]] = []
        self._structured_calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> MockAIResponse:
        self.invocations.append(messages)
        return next(self._responses)

    def bind_tools(self, tools: list[Any]) -> MockModel:
        return self

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


# ---------------------------------------------------------------------------
# Mock tool
# ---------------------------------------------------------------------------


class MockTool:
    """Mock tool with a fixed result."""

    def __init__(self, name: str, result: str = "tool result") -> None:
        self.name = name
        self.description = f"Mock {name} tool"
        self._result = result
        self.invocations: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.invocations.append(args)
        return self._result

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


# ---------------------------------------------------------------------------
# Fake agent loops
# ---------------------------------------------------------------------------


class FakeAgentLoop:
    """Controllable AgentLoop implementation for testing.

    Supports both str and list[Message] input types.
    Emits a configurable number of llm_end events, then a complete event
    carrying an AgentResult with the given output.
    """

    def __init__(
        self,
        steps: int = 1,
        result_output: Any = "done",
        delay: float = 0,
    ) -> None:
        self._steps = steps
        self._result_output = result_output
        self._delay = delay
        self._inbox: list[str] = []
        self._cancelled = False

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data.get("result")  # type: ignore[return-value]
        raise RuntimeError("stream ended without complete event")

    async def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id: str = config.metadata.get("agent_id", "")  # type: ignore[assignment]
        for step in range(self._steps):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield AgentEvent(type="llm_end", agent_id=agent_id, step=step)
        result = AgentResult(
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            output=self._result_output,
            steps=self._steps,
        )
        yield AgentEvent(
            type="complete", agent_id=agent_id, data={"result": result}
        )


class NeverEndingLoop:
    """A loop that never completes -- hangs until cancelled."""

    def __init__(self) -> None:
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        self._inbox.append(message)

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                return event.data.get("result")  # type: ignore[return-value]
        raise RuntimeError("stream ended without complete event")

    async def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> Any:
        config = config or RunConfig()
        agent_id: str = config.metadata.get("agent_id", "")  # type: ignore[assignment]
        yield AgentEvent(type="llm_start", agent_id=agent_id, step=0)
        try:
            await asyncio.sleep(3600)  # effectively forever
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------


class MockEventHandler:
    """Collects all events for assertion."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------


class FakeWorkerFactory:
    """Mock factory that produces FakeAgentLoop instances.

    Tracks create_worker calls for assertion.
    """

    _UNSET: Any = object()

    def __init__(
        self,
        loop: FakeAgentLoop | None = None,
        result_output: Any = _UNSET,
    ) -> None:
        self._loop = loop
        self._result_output = result_output
        self.create_calls: list[tuple[str, str, str | None]] = []

    def create_worker(
        self,
        agent_id: str,
        task_type: str,
        *,
        task_id: str | None = None,
    ) -> FakeAgentLoop:
        self.create_calls.append((agent_id, task_type, task_id))
        if self._loop is not None:
            return self._loop
        if self._result_output is not self._UNSET:
            return FakeAgentLoop(result_output=self._result_output)
        return FakeAgentLoop()
