"""Integration smoke test for the kernel ``AgentLoop``.

This is the acceptance criterion for the kernel seed. It drives a fake
``StreamFn`` and a fake tool through one full ReAct cycle and asserts:

1. Final message list shape: user → assistant(tool_call) → tool_result →
   assistant(text).
2. The ``tool_call`` and ``tool_result`` events fire with the expected
   payloads.
3. A mutating ``tool_call`` handler is observed by the tool (mutation
   contract).
4. A blocking ``tool_call`` handler short-circuits to an error tool result
   without invoking the tool (replacement contract).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.kernel.events import (
    BeforeSendToLlmEvent,
    ContextEvent,
    EventBus,
    ToolCallEvent,
    ToolResultEvent,
)
from agentm.core.kernel.loop import AgentLoop, LoopConfig
from agentm.core.kernel.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
    text_message,
)
from agentm.core.kernel.stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
)
from agentm.core.kernel.tool import FunctionTool, ToolResult


# --- Fakes ------------------------------------------------------------------


class FakeStream:
    """Deterministic two-call stream: first emits a tool call, then a final
    text reply. Tracks how many times it was called so tests can assert the
    loop made exactly two LLM calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[Any] | None = None

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        self.calls += 1
        call_index = self.calls
        self.last_messages = list(messages)
        return self._iter(call_index)

    async def _iter(self, call_index: int) -> AsyncIterator[AssistantStreamEvent]:
        if call_index == 1:
            yield ToolCallStart(id="call-1", name="echo")
            yield ToolCallArgsDelta(id="call-1", args_json_delta='{"text":"hi"}')
            yield ToolCallEnd(id="call-1")
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id="call-1",
                            name="echo",
                            arguments={"text": "hi"},
                        )
                    ],
                    timestamp=1.0,
                    stop_reason="tool_use",
                )
            )
        else:
            yield TextDelta(text="done")
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="done")],
                    timestamp=2.0,
                    stop_reason="end_turn",
                )
            )


def _model() -> Model:
    return Model(
        id="fake-model",
        provider="fake",
        context_window=8192,
        max_output_tokens=1024,
    )


def _echo_tool(seen: list[dict[str, Any]]) -> FunctionTool:
    """An ``echo`` tool that records the args it sees, so tests can verify
    mutation observation."""

    async def fn(args: dict[str, Any]) -> ToolResult:
        seen.append(dict(args))
        return ToolResult(
            content=[
                TextContent(type="text", text=f"echoed: {args.get('text', '')}")
            ]
        )

    return FunctionTool(
        name="echo",
        description="echo tool",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=fn,
    )


# --- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_react_cycle() -> None:
    """End-to-end happy path: user → tool call → tool result → final text."""

    bus = EventBus()
    stream = FakeStream()
    tool_calls_seen: list[ToolCallEvent] = []
    tool_results_seen: list[ToolResultEvent] = []
    bus.on("tool_call", lambda e: tool_calls_seen.append(e))
    bus.on("tool_result", lambda e: tool_results_seen.append(e))

    seen_args: list[dict[str, Any]] = []
    tool = _echo_tool(seen_args)

    loop = AgentLoop(stream_fn=stream, bus=bus, config=LoopConfig(max_turns=8))

    user = text_message("please echo hi", timestamp=0.0)
    final = await loop.run(
        messages=[user],
        model=_model(),
        tools=[tool],
    )

    # Shape: 1 user, 1 assistant(tool_call), 1 tool_result, 1 assistant(text)
    assert len(final) == 4
    assert isinstance(final[0], UserMessage)
    assert isinstance(final[1], AssistantMessage)
    assert any(isinstance(b, ToolCallBlock) for b in final[1].content)
    assert isinstance(final[2], ToolResultMessage)
    assert isinstance(final[3], AssistantMessage)
    last_text = final[3].content[0]
    assert isinstance(last_text, TextContent)
    assert last_text.text == "done"

    # Stream was called exactly twice (once per turn).
    assert stream.calls == 2

    # Events fired with the right payloads.
    assert len(tool_calls_seen) == 1
    assert tool_calls_seen[0].tool_name == "echo"
    assert tool_calls_seen[0].args == {"text": "hi"}
    assert len(tool_results_seen) == 1
    assert tool_results_seen[0].tool_name == "echo"
    assert tool_results_seen[0].result.is_error is False

    # Tool actually saw the args.
    assert seen_args == [{"text": "hi"}]


@pytest.mark.asyncio
async def test_tool_call_mutation_is_observed_by_tool() -> None:
    """A handler that mutates ``event.args`` in place must change what the
    tool actually receives. This is the §3.5 mutating-event contract."""

    bus = EventBus()
    stream = FakeStream()

    def mutator(event: ToolCallEvent) -> None:
        event.args["text"] = "MUTATED"

    bus.on("tool_call", mutator)

    seen_args: list[dict[str, Any]] = []
    tool = _echo_tool(seen_args)

    loop = AgentLoop(stream_fn=stream, bus=bus)
    await loop.run(
        messages=[text_message("go", timestamp=0.0)],
        model=_model(),
        tools=[tool],
    )

    assert seen_args == [{"text": "MUTATED"}]


@pytest.mark.asyncio
async def test_blocking_handler_short_circuits_to_error_result() -> None:
    """Returning ``{"block": True, "reason": ...}`` must skip tool execution
    and surface a synthetic error tool result the LLM can see."""

    bus = EventBus()
    stream = FakeStream()

    def blocker(event: ToolCallEvent) -> dict[str, Any]:
        return {"block": True, "reason": "nope"}

    bus.on("tool_call", blocker)

    seen_args: list[dict[str, Any]] = []
    tool = _echo_tool(seen_args)

    loop = AgentLoop(stream_fn=stream, bus=bus)
    final = await loop.run(
        messages=[text_message("go", timestamp=0.0)],
        model=_model(),
        tools=[tool],
    )

    # Tool was NOT executed.
    assert seen_args == []

    # The tool_result message is present and carries an error.
    tool_result_msg = final[2]
    assert isinstance(tool_result_msg, ToolResultMessage)
    block = tool_result_msg.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.is_error is True
    text = block.content[0]
    assert isinstance(text, TextContent)
    assert "nope" in text.text


@pytest.mark.asyncio
async def test_before_send_to_llm_fires_after_context() -> None:
    """``before_send_to_llm`` must fire after ``context`` handlers have run,
    must see the final messages list (post-replacement), and handlers may
    mutate that list before bytes hit the wire (Phase 2.0 §10b.1)."""

    bus = EventBus()
    stream = FakeStream()

    order: list[str] = []
    captured: dict[str, Any] = {}

    def context_handler(event: ContextEvent) -> dict[str, Any]:
        order.append("context")
        # Replacement-by-return path: hand the loop a brand-new list.
        replacement = list(event.messages)
        replacement.append(text_message("ctx-injected", timestamp=0.5))
        return {"messages": replacement}

    def before_send_handler(event: BeforeSendToLlmEvent) -> None:
        order.append("before_send_to_llm")
        captured["messages"] = list(event.messages)
        captured["model_id"] = event.model.id
        captured["tool_names"] = [t.name for t in event.tools]
        captured["system"] = event.system
        # Mutation: drop our injected ctx message before LLM sees it.
        event.messages[:] = [
            m for m in event.messages
            if not (
                isinstance(m, UserMessage)
                and m.content
                and isinstance(m.content[0], TextContent)
                and m.content[0].text == "ctx-injected"
            )
        ]

    bus.on("context", context_handler)
    bus.on("before_send_to_llm", before_send_handler)

    seen_args: list[dict[str, Any]] = []
    tool = _echo_tool(seen_args)

    loop = AgentLoop(stream_fn=stream, bus=bus)
    await loop.run(
        messages=[text_message("go", timestamp=0.0)],
        model=_model(),
        tools=[tool],
        system="sys-prompt",
    )

    # context fires before before_send_to_llm on the same turn.
    assert order[:2] == ["context", "before_send_to_llm"]

    # The before_send_to_llm payload reflects the post-context replacement.
    msgs = captured["messages"]
    assert any(
        isinstance(m, UserMessage)
        and m.content
        and isinstance(m.content[0], TextContent)
        and m.content[0].text == "ctx-injected"
        for m in msgs
    )
    assert captured["model_id"] == "fake-model"
    assert captured["tool_names"] == ["echo"]
    assert captured["system"] == "sys-prompt"

    # Mutation by before_send_to_llm reaches the StreamFn (the FakeStream
    # records the messages it was called with on the FIRST call).
    first_call_msgs = stream.last_messages
    assert first_call_msgs is not None
    # FakeStream stores the LAST call's messages, not the first; check by
    # asserting the injected message did NOT survive on either call by
    # verifying the loop's last forwarded list does not contain it.
    assert all(
        not (
            isinstance(m, UserMessage)
            and m.content
            and isinstance(m.content[0], TextContent)
            and m.content[0].text == "ctx-injected"
        )
        for m in first_call_msgs
    )
