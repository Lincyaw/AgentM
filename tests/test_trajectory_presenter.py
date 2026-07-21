from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.termination import ModelEndTurn, SignalAborted
from agentm.core.abi.trajectory import Outcome, ToolRecord, Turn
from agentm.core.abi.trigger import ContinueTrigger, UserInput
from agentm.presenter.trajectory import build_trace_snapshot


def test_turn_cause_is_not_copied_to_every_trajectory_row() -> None:
    call = ToolCallBlock(
        type="tool_call",
        id="read-1",
        name="read",
        arguments={"path": "src/example.py"},
    )
    result = ToolResultBlock(
        type="tool_result",
        tool_call_id=call.id,
        content=(TextContent(type="text", text="file contents"),),
    )
    turn = Turn(
        index=0,
        id="interrupted-turn",
        run_id="run-1",
        run_step=0,
        trigger=UserInput(
            content=(TextContent(type="text", text="inspect the repository"),)
        ),
        response=AssistantMessage(
            role="assistant",
            content=(call,),
            timestamp=1.0,
            stop_reason="tool_use",
        ),
        tool_results=(ToolRecord(call=call, result=result),),
        outcome=Outcome(cause=SignalAborted(reason="submit_interrupt")),
        timestamp=2.0,
    )

    snapshot = build_trace_snapshot("session-1", (turn,))

    assert snapshot.turns[0].cause == "SignalAborted"
    assert snapshot.turns[0].run_id == turn.run_id
    assert snapshot.turns[0].run_step == turn.run_step
    control = next(row for row in snapshot.rows if row.kind == "control")
    assert control.cause == "SignalAborted"
    historical_rows = tuple(row for row in snapshot.rows if row.kind != "control")
    assert historical_rows
    assert all(row.run_id == turn.run_id for row in historical_rows)
    assert all(row.run_step == turn.run_step for row in historical_rows)
    assert all(row.cause is None for row in historical_rows)
    assert all(row.status == "committed" for row in historical_rows)


def test_internal_continuation_is_not_rendered_as_a_user_message() -> None:
    turn = Turn(
        index=1,
        id="continuation-turn",
        run_id="run-1",
        run_step=1,
        trigger=ContinueTrigger(),
        response=AssistantMessage(
            role="assistant",
            content=(TextContent(type="text", text="done"),),
            timestamp=1.0,
            stop_reason="end_turn",
        ),
        tool_results=(),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=2.0,
    )

    snapshot = build_trace_snapshot("session-1", (turn,))

    assert not any(row.kind in {"user", "trigger"} for row in snapshot.rows)
    assert snapshot.turns[0].trigger == ""
