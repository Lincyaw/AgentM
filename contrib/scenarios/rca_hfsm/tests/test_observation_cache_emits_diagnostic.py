"""Acceptance #5 — cache emits a ``tool_call_cached`` diagnostic on hit.

The cache atom is wired to publish a ``DiagnosticEvent`` whose message
contains the literal ``"tool_call_cached"`` sentinel every time it
short-circuits a tool call. Observability sinks subscribe to
``DiagnosticEvent.CHANNEL`` and persist the event into the trace JSONL;
this test asserts the publish step (the sink is an orthogonal builtin).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import AgentStartEvent, FunctionTool, ToolResult
from agentm.core.abi.events import DiagnosticEvent
from agentm.core.abi.messages import TextContent

from tests._gate_fixtures import install_full_stack


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_cache_hit_emits_tool_call_cached_diagnostic() -> None:
    api, _, _ = install_full_stack()

    async def _execute(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text=f"q={args.get('q')}")])

    api.register_tool(
        FunctionTool(
            name="stub_idem",
            description="t",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            fn=_execute,
            metadata={"idempotent": True},
        )
    )
    api.events.fire_handlers(
        AgentStartEvent.CHANNEL, AgentStartEvent(messages=[])
    )
    tool = next(t for t in api.tools if t.name == "stub_idem")

    _run(tool.execute({"q": "alpha"}))  # miss — no diagnostic
    _run(tool.execute({"q": "alpha"}))  # hit — diagnostic fires

    cached_events = [
        event
        for channel, event in api.events.emitted
        if channel == DiagnosticEvent.CHANNEL
        and isinstance(event, DiagnosticEvent)
        and "tool_call_cached" in event.message
    ]
    assert len(cached_events) == 1
    diag = cached_events[0]
    assert diag.source == "rca_observation_cache"
    assert diag.level == "info"


def test_cache_miss_does_not_emit_diagnostic() -> None:
    api, _, _ = install_full_stack()

    async def _execute(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="x")])

    api.register_tool(
        FunctionTool(
            name="stub_miss",
            description="t",
            parameters={"type": "object", "properties": {}},
            fn=_execute,
            metadata={"idempotent": True},
        )
    )
    api.events.fire_handlers(
        AgentStartEvent.CHANNEL, AgentStartEvent(messages=[])
    )
    tool = next(t for t in api.tools if t.name == "stub_miss")

    _run(tool.execute({}))  # cold miss

    cached_events = [
        event
        for channel, event in api.events.emitted
        if channel == DiagnosticEvent.CHANNEL
        and isinstance(event, DiagnosticEvent)
        and "tool_call_cached" in event.message
    ]
    assert cached_events == []
