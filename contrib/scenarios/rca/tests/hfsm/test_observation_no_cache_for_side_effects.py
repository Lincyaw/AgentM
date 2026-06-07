"""Counter-acceptance #5 — non-idempotent tools are never memoised.

A stub tool registered with ``metadata['idempotent']=False`` must execute
on every invocation, even when called twice with identical args. The
cache atom's wrap-at-agent-start filter is the gate; the test fires
``AgentStartEvent`` to exercise the same code path as the positive test.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import AgentStartEvent, FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent

from tests.hfsm._gate_fixtures import install_full_stack


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_non_idempotent_tool_is_never_cached() -> None:
    api, _, _ = install_full_stack()
    call_count = [0]

    async def _execute(args: dict[str, Any]) -> ToolResult:
        call_count[0] += 1
        return ToolResult(
            content=[
                TextContent(
                    type="text", text=f"call #{call_count[0]} q={args.get('q')}"
                )
            ]
        )

    api.register_tool(
        FunctionTool(
            name="stub_side_effect",
            description="test",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            fn=_execute,
            metadata={"idempotent": False},
        )
    )
    api.events.fire_handlers(
        AgentStartEvent.CHANNEL, AgentStartEvent(messages=[])
    )

    tool = next(t for t in api.tools if t.name == "stub_side_effect")

    _run(tool.execute({"q": "alpha"}))
    _run(tool.execute({"q": "alpha"}))
    _run(tool.execute({"q": "alpha"}))

    # Every call advances the counter; the cache wrapper never intercepted.
    assert call_count[0] == 3


def test_tool_without_metadata_idempotent_field_is_not_cached() -> None:
    """A tool that omits the metadata flag entirely (default-false) is
    treated as side-effecting. This matches the design's "opt in" stance:
    tools must declare they're safe to memoise."""

    api, _, _ = install_full_stack()
    call_count = [0]

    async def _execute(args: dict[str, Any]) -> ToolResult:
        call_count[0] += 1
        return ToolResult(
            content=[TextContent(type="text", text=f"n={call_count[0]}")]
        )

    api.register_tool(
        FunctionTool(
            name="stub_no_metadata",
            description="test",
            parameters={"type": "object", "properties": {}},
            fn=_execute,
            metadata={},  # No idempotent key at all.
        )
    )
    api.events.fire_handlers(
        AgentStartEvent.CHANNEL, AgentStartEvent(messages=[])
    )

    tool = next(t for t in api.tools if t.name == "stub_no_metadata")
    _run(tool.execute({}))
    _run(tool.execute({}))

    assert call_count[0] == 2
