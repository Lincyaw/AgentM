"""``trace_query`` against a parent trajectory that has real tool calls in it.

The atom reads its *parent* session, which is the one thing that makes it hard
to exercise: on a root session every query correctly answers nothing, and a
test that never spawned a child would pass while the tools were unusable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm import ExtensionSpec
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
    freeze_json,
)
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, ToolRecord, Turn, TurnMeta
from agentm.core.abi.trigger import UserInput
from agentm.extensions.builtin import trace_query


def _turn_with_a_tool_call() -> Turn:
    """A turn shaped the way the runtime holds one.

    ``freeze_json`` is the point: every message the runtime keeps has been
    through it, so a tool call's ``arguments`` is a ``MappingProxyType``. Built
    from a plain dict, this test would pass against code that cannot read a
    single real trajectory.
    """

    call = ToolCallBlock(
        type="tool_call",
        id="call-1",
        name="bash",
        arguments=freeze_json({"cmd": "echo one", "timeout": 10}),
    )
    return Turn(
        index=0,
        id="turn-1",
        run_id="run-1",
        run_step=0,
        trigger=UserInput(content=(TextContent(type="text", text="do it"),)),
        response=AssistantMessage(
            role="assistant",
            content=(TextContent(type="text", text="running it"), call),
            timestamp=0.0,
        ),
        tool_results=(
            ToolRecord(
                call=call,
                result=ToolResultBlock(
                    type="tool_result",
                    tool_call_id="call-1",
                    content=(TextContent(type="text", text="one"),),
                ),
            ),
        ),
        outcome=Outcome(cause=ModelEndTurn()),
        meta=TurnMeta(model_id="probe-model"),
        timestamp=0.0,
    )


class _NoParent:
    """An api with no parent, so the runtime keeps the turns it was handed.

    ``_refresh_turns`` re-reads the parent from the store on every call and
    falls back to what it holds when there is none, which is what lets these
    checks fix the trajectory instead of standing up two sessions and a store.
    """

    class ctx:  # noqa: N801 - stands in for SessionContext
        parent_session_id = None

    store = None


def _runtime(turns: list[Turn] | None = None) -> trace_query._TraceQueryRuntime:
    return trace_query._TraceQueryRuntime(
        _NoParent(), [_turn_with_a_tool_call()] if turns is None else turns
    )


@pytest.mark.asyncio
async def test_a_tool_call_can_actually_be_reported(tmp_path: Path) -> None:
    """Tool arguments arrive frozen, and ``json.dumps`` refuses a mappingproxy.

    Every real tool call this atom was asked about raised ``TypeError: Object of
    type mappingproxy is not JSON serializable`` -- which is every call it
    exists to report, on the one path it was written for: the ``goal`` checker
    inspecting what its parent did.
    """

    del tmp_path
    runtime = _runtime()

    reported = "".join(
        block.text
        for block in (await runtime.get_tool_calls({})).content
        if isinstance(block, TextContent)
    )
    assert "Tool calls: 1 total" in reported
    assert '"cmd": "echo one"' in reported

    read = "".join(
        block.text
        for block in (await runtime.read_turn({})).content
        if isinstance(block, TextContent)
    )
    assert "bash" in read


@pytest.mark.asyncio
async def test_an_argument_the_model_got_wrong_is_said_not_dropped() -> None:
    """A filter that is silently discarded answers with everything.

    These tools are called by a model, so a mistyped argument name used to
    return an unfiltered result that the caller had every reason to read as
    filtered. The orchestrator turns the refusal into an error result the model
    can act on, which is the whole difference between wrong and unknown.
    """

    runtime = _runtime()
    filtered = "".join(
        block.text
        for block in (await runtime.get_tool_calls({"tool_name": "nothing"})).content
        if isinstance(block, TextContent)
    )
    assert "Tool calls: 0 total" in filtered

    with pytest.raises(Exception, match="extra_forbidden"):
        await runtime.get_tool_calls({"tool": "bash"})


@pytest.mark.asyncio
async def test_an_empty_page_does_not_report_a_negative_range() -> None:
    """``showing 0--1`` is what an empty slice rendered."""

    runtime = _runtime([])
    empty = "".join(
        block.text
        for block in (await runtime.read_turn({})).content
        if isinstance(block, TextContent)
    )
    assert empty.strip() == "Messages: 0 total"


@pytest.mark.asyncio
async def test_the_atom_installs_and_reverts(tmp_path: Path) -> None:
    """The tools reach a session, which the unit checks above do not witness."""

    from agentm.testing import assert_revertible, probe_session

    async with probe_session(str(tmp_path)) as session:
        await assert_revertible(
            session, ExtensionSpec.from_module("agentm.extensions.builtin.trace_query")
        )
