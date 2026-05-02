"""Tests for the per-turn decision protocol.

These cover the loop-defining failure positions for the redesigned
``decide_turn_action`` channel: if any of these contracts break, the SDK
boundary every extension depends on (Inject vs Stop vs Step lattice, final
overrides ignored, provider-protocol violations distinguished from clean
end_turns) silently regresses for everyone.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import (
    AgentLoop,
    AssistantMessage,
    DecideTurnActionEvent,
    EventBus,
    FunctionTool,
    Inject,
    LoopAction,
    LoopConfig,
    MaxTurnsExhausted,
    MessageEnd,
    Model,
    ModelEndTurn,
    ProviderProtocolViolation,
    SignalAborted,
    Step,
    Stop,
    TextContent,
    ToolCallBlock,
    ToolResult,
    ToolTerminate,
    ToolTerminated,
    text_message,
)
from agentm.core.abi.loop import (
    _default_action_with_names,
    _resolve_action,
)
from agentm.core.abi.tool import ToolContinue


class _ScriptedStream:
    def __init__(self, scripted: list[AssistantMessage]) -> None:
        self._scripted = scripted
        self.calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[MessageEnd]:
        del messages, model, tools, system, signal, thinking
        idx = self.calls
        self.calls += 1
        return self._iter(self._scripted[idx])

    async def _iter(self, message: AssistantMessage) -> AsyncIterator[MessageEnd]:
        yield MessageEnd(message=message)


def _model() -> Model:
    return Model(
        id="test", provider="fake", context_window=4096, max_output_tokens=512
    )


@pytest.mark.asyncio
async def test_inject_keeps_loop_alive_and_appends_messages() -> None:
    """An Inject return on a default Stop must keep the loop alive AND
    cause the messages to be visible on the next turn."""

    stream = _ScriptedStream(
        [
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="wrapping up")],
                timestamp=1.0,
                stop_reason="end_turn",
            ),
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="final answer")],
                timestamp=2.0,
                stop_reason="end_turn",
            ),
        ]
    )
    bus = EventBus()
    decisions: list[LoopAction] = []
    end_causes: list[Any] = []

    async def on_decide(event: DecideTurnActionEvent) -> LoopAction | None:
        decisions.append(event.observation.default_action)
        if len(decisions) == 1:
            return Inject(
                messages=[text_message("<subagent_result>ready</subagent_result>")]
            )
        return None

    bus.on("decide_turn_action", on_decide)
    bus.on("agent_end", lambda e: end_causes.append(e.cause))

    loop = AgentLoop(stream_fn=stream, bus=bus)
    messages = await loop.run(
        messages=[text_message("start")],
        model=_model(),
        tools=[],
    )

    # Two LLM calls: first turn injected, second turn finished cleanly.
    assert stream.calls == 2
    assert len(decisions) == 2
    assert all(isinstance(d, Stop) and isinstance(d.cause, ModelEndTurn) for d in decisions)
    assert len(end_causes) == 1 and isinstance(end_causes[0], ModelEndTurn)
    # The injected message must be present at index 2 (after start + first
    # assistant + injected user).
    injected = messages[2]
    assert injected.role == "user"
    injected_block = injected.content[0]
    assert isinstance(injected_block, TextContent)
    assert injected_block.text == "<subagent_result>ready</subagent_result>"
    final = messages[-1]
    assert final.role == "assistant"
    final_block = final.content[0]
    assert isinstance(final_block, TextContent)
    assert final_block.text == "final answer"


@pytest.mark.asyncio
async def test_max_turns_is_final_and_inject_overrides_are_ignored() -> None:
    """``MaxTurnsExhausted.final`` must shadow any extension Inject — a
    handler returning Inject should NOT keep the loop alive past max_turns,
    but the hook should still fire once for observability."""

    stream = _ScriptedStream(
        [
            AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="t1",
                        name="noop",
                        arguments={},
                    )
                ],
                timestamp=1.0,
                stop_reason="tool_use",
            )
        ]
    )
    bus = EventBus()
    seen_defaults: list[LoopAction] = []
    end_causes: list[Any] = []

    async def on_decide(event: DecideTurnActionEvent) -> LoopAction:
        seen_defaults.append(event.observation.default_action)
        return Inject(messages=[text_message("trying to keep it alive")])

    bus.on("decide_turn_action", on_decide)
    bus.on("agent_end", lambda e: end_causes.append(e.cause))

    async def _noop(_args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="ok")])

    loop = AgentLoop(stream_fn=stream, bus=bus, config=LoopConfig(max_turns=1))
    await loop.run(
        messages=[text_message("start")],
        model=_model(),
        tools=[
            FunctionTool(
                name="noop",
                description="No-op tool",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                fn=_noop,
            )
        ],
    )

    # Exactly two decisions: one for the in-loop turn (Step), one for the
    # max-turns terminal observation (Stop with MaxTurnsExhausted).
    assert len(seen_defaults) == 2
    assert isinstance(seen_defaults[0], Step)
    assert isinstance(seen_defaults[1], Stop) and isinstance(
        seen_defaults[1].cause, MaxTurnsExhausted
    )
    assert len(end_causes) == 1 and isinstance(end_causes[0], MaxTurnsExhausted)


def test_default_action_distinguishes_protocol_violation_from_end_turn() -> None:
    """Provider says ``tool_use`` but no tool_calls extracted → must surface
    as :class:`ProviderProtocolViolation`, not silently mapped to
    :class:`ModelEndTurn`. This is the bug that justifies the rich-cause
    redesign in the first place."""

    msg_violation = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="weird")],
        timestamp=1.0,
        stop_reason="tool_use",
    )
    msg_clean = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=1.0,
        stop_reason="end_turn",
    )

    violation = _default_action_with_names(msg_violation, [])
    clean = _default_action_with_names(msg_clean, [])

    assert isinstance(violation, Stop)
    assert isinstance(violation.cause, ProviderProtocolViolation)
    assert isinstance(clean, Stop)
    assert isinstance(clean.cause, ModelEndTurn)


def test_default_action_maps_tool_terminate_to_tool_terminated_cause() -> None:
    """A tool returning :class:`ToolTerminate` must yield
    ``Stop(ToolTerminated(tool_name=..., reason=...))`` — the kernel must
    plumb the tool's identity all the way through."""

    msg = AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call", id="t1", name="submit_final_report", arguments={}
            )
        ],
        timestamp=1.0,
        stop_reason="tool_use",
    )
    outcome = ToolTerminate(
        result=ToolResult(content=[TextContent(type="text", text="done")]),
        reason="rca-final-report-submitted",
    )
    action = _default_action_with_names(msg, [("submit_final_report", outcome)])

    assert isinstance(action, Stop)
    assert isinstance(action.cause, ToolTerminated)
    assert action.cause.tool_name == "submit_final_report"
    assert action.cause.reason == "rca-final-report-submitted"


def test_resolve_action_concatenates_multiple_inject_returns_in_order() -> None:
    """When several handlers return :class:`Inject`, the resolved action's
    messages must be the concatenation in registration order — extensions
    stack rather than fight, and ordering is observable."""

    default = Stop(ModelEndTurn())
    msg_a = text_message("from-a")
    msg_b = text_message("from-b")
    msg_c = text_message("from-c")

    resolved = _resolve_action(
        default,
        [
            Inject(messages=[msg_a]),
            None,
            Inject(messages=[msg_b, msg_c]),
        ],
    )

    assert isinstance(resolved, Inject)
    texts: list[str] = []
    for m in resolved.messages:
        block = m.content[0]
        assert isinstance(block, TextContent)
        texts.append(block.text)
    assert texts == ["from-a", "from-b", "from-c"]


def test_resolve_action_stop_overrides_step_default_when_handler_asks() -> None:
    """A handler returning :class:`Stop` must override a ``Step`` default —
    extensions can choose to terminate even when more tool calls were
    pending. Inject still beats Stop when both are present."""

    default = Step()
    only_stop = _resolve_action(
        default, [Stop(ModelEndTurn())]
    )
    inject_wins = _resolve_action(
        default, [Stop(ModelEndTurn()), Inject(messages=[text_message("x")])]
    )

    assert isinstance(only_stop, Stop)
    assert isinstance(only_stop.cause, ModelEndTurn)
    assert isinstance(inject_wins, Inject)


@pytest.mark.asyncio
async def test_signal_mid_tool_emits_agent_end_exactly_once() -> None:
    """Cooperative signal abort between LLM turn and tool execution must
    route through ``_terminate`` so ``agent_end`` and ``decide_turn_action``
    each fire exactly once. Regression guard: a previous draft of the
    sum-type rewrite double-emitted ``agent_end`` (once via
    ``_finish_with_cause`` inside the tool loop, once via the next
    iteration's ``_terminate``), which silently corrupts trajectory rollups
    and the evolution-substrate indexer."""

    stream = _ScriptedStream(
        [
            AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="t1",
                        name="noop",
                        arguments={},
                    )
                ],
                timestamp=1.0,
                stop_reason="tool_use",
            )
        ]
    )
    bus = EventBus()
    signal = asyncio.Event()
    decisions: list[Any] = []
    end_causes: list[Any] = []

    async def trip_signal_after_turn(_event: Any) -> None:
        signal.set()

    bus.on("turn_end", trip_signal_after_turn)
    bus.on(
        "decide_turn_action",
        lambda e: decisions.append(e.observation.default_action),
    )
    bus.on("agent_end", lambda e: end_causes.append(e.cause))

    async def _noop(_args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="ok")])

    loop = AgentLoop(stream_fn=stream, bus=bus)
    await loop.run(
        messages=[text_message("start")],
        model=_model(),
        tools=[
            FunctionTool(
                name="noop",
                description="No-op tool",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                fn=_noop,
            )
        ],
        signal=signal,
    )

    assert len(end_causes) == 1
    assert isinstance(end_causes[0], SignalAborted)
    assert len(decisions) == 1
    default = decisions[0]
    assert isinstance(default, Stop)
    assert isinstance(default.cause, SignalAborted)


def test_bare_tool_result_normalizes_to_tool_continue() -> None:
    """A tool returning a bare :class:`ToolResult` (the legacy/simple shape)
    must be treated as :class:`ToolContinue` — the loop must NOT confuse
    "tool returned successfully" with "tool wants to terminate"."""

    msg = AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call", id="t1", name="some_tool", arguments={}
            )
        ],
        timestamp=1.0,
        stop_reason="tool_use",
    )
    bare_result = ToolResult(content=[TextContent(type="text", text="ok")])
    paired: list[tuple[str, Any]] = [
        ("some_tool", ToolContinue(result=bare_result))
    ]

    action = _default_action_with_names(msg, paired)

    assert isinstance(action, Step)
