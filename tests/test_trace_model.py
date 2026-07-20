from __future__ import annotations

from agentm.presenter.trajectory import (
    build_trace_snapshot,
    default_trace_view_registry,
    filter_trace_rows,
    parse_trace_query,
)
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, Round, ToolRecord, Turn, TurnMeta
from agentm.core.abi.trigger import UserInput


def _sample_turn(*, tool_error: bool = True) -> Turn:
    call = ToolCallBlock(
        type="tool_call",
        id="call-1",
        name="bash",
        arguments={"cmd": "echo Set-Cookie"},
    )
    result = ToolResultBlock(
        type="tool_result",
        tool_call_id="call-1",
        content=(TextContent(type="text", text="Set-Cookie preserved"),),
        is_error=tool_error,
    )
    return Turn(
        index=0,
        id="turn-1",
        trigger=UserInput(
            content=(TextContent(type="text", text="fix oauth response"),)
        ),
        rounds=(
            Round(
                response=AssistantMessage(
                    role="assistant",
                    content=(TextContent(type="text", text="I will inspect it"), call),
                    timestamp=2.0,
                ),
                tool_results=(ToolRecord(call=call, result=result),),
            ),
        ),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=3.0,
        meta=TurnMeta(
            total_input_tokens=150,
            total_output_tokens=25,
            cache_read_tokens=40,
            cache_write_tokens=7,
            model_id="test-model",
            system_prompt="system prompt",
        ),
    )


def test_build_trace_snapshot_flattens_messages_and_commit_signal() -> None:
    snapshot = build_trace_snapshot("session-1", [_sample_turn()])

    assert snapshot.status_label == "complete"
    assert snapshot.metrics.committed_turns == 1
    assert snapshot.metrics.tool_errors == 1
    assert snapshot.metrics.cache_write_tokens == 7
    assert [row.kind for row in snapshot.rows] == [
        "system",
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "control",
    ]
    assert snapshot.rows[-1].title == "TURN COMMITTED"
    assert "ModelEndTurn" in snapshot.rows[-1].content


def test_trace_query_filters_tool_errors_and_text() -> None:
    snapshot = build_trace_snapshot("session-1", [_sample_turn()])
    query = parse_trace_query('tool:bash error text:"Set-Cookie"')

    rows = filter_trace_rows(snapshot.rows, query)

    assert [row.kind for row in rows] == ["tool_result"]
    assert rows[0].tool_name == "bash"
    assert rows[0].is_error is True


def test_trace_query_filters_status_cause_and_tokens() -> None:
    snapshot = build_trace_snapshot("session-1", [_sample_turn(tool_error=False)])
    query = parse_trace_query("status:committed cause:ModelEndTurn tokens>100")

    rows = filter_trace_rows(snapshot.rows, query)

    assert rows
    assert {row.status for row in rows} == {"committed"}
    assert {row.cause for row in rows} == {"ModelEndTurn"}


def test_metrics_view_builds_summary_rows() -> None:
    snapshot = build_trace_snapshot("session-1", [_sample_turn()])
    view = (
        default_trace_view_registry()
        .get("metrics")
        .build(
            snapshot,
            parse_trace_query(""),
        )
    )

    assert {row.title for row in view.rows} == {"Turns", "Tokens", "Tools", "Rows"}
    assert any("input_tokens: 150" in row.content for row in view.rows)
