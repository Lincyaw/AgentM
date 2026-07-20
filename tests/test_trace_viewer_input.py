from __future__ import annotations

import os

from agentm.cli import _trace_viewer as viewer
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, Round, ToolRecord, Turn, TurnMeta
from agentm.core.abi.trigger import UserInput
from agentm.presenter.trajectory import build_trace_snapshot


def _read_key_from_bytes(data: bytes) -> str:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)
        os.close(write_fd)
        write_fd = -1
        return viewer._read_key(read_fd)
    finally:
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)


def test_trace_viewer_reads_supported_key_sequences() -> None:
    cases = {
        b"\x1b": viewer._KEY_ESC,
        b"\x1b[A": viewer._KEY_UP,
        b"\x1b[B": viewer._KEY_DOWN,
        b"\x1b[5~": viewer._KEY_PAGE_UP,
        b"\x1b[6~": viewer._KEY_PAGE_DOWN,
        b"\x1b[<64;10;20M": viewer._KEY_SCROLL_UP,
        b"\x1b[<65;10;20M": viewer._KEY_SCROLL_DOWN,
        b"\x1b[<0;10;20M": viewer._KEY_IGNORE,
        b"\x1b[?1006h": viewer._KEY_IGNORE,
    }

    for data, expected in cases.items():
        assert _read_key_from_bytes(data) == expected


def test_trace_viewer_mouse_wheel_scrolls_expanded_viewport() -> None:
    tui = viewer.TraceViewer.__new__(viewer.TraceViewer)
    tui._views = []
    tui._cursor = 0
    tui._items = [
        viewer._ViewItem(
            msg=viewer._Message(role="assistant", content=""),
            body_lines=[f"line {i}" for i in range(30)],
            expanded=True,
        )
    ]
    tui._scroll = 0
    tui._manual_scroll = False
    tui._term_size = lambda: (80, 10)

    tui._handle_expanded_key(viewer._KEY_SCROLL_DOWN)
    assert tui._scroll == viewer._WHEEL_SCROLL_LINES
    assert tui._manual_scroll is True

    tui._handle_expanded_key(viewer._KEY_SCROLL_UP)
    assert tui._scroll == 0


def test_trace_viewer_builds_views_from_trace_snapshot_rows() -> None:
    call = ToolCallBlock(
        type="tool_call",
        id="call-1",
        name="bash",
        arguments={"cmd": "echo hi"},
    )
    result = ToolResultBlock(
        type="tool_result",
        tool_call_id="call-1",
        content=(TextContent(type="text", text="boom"),),
        is_error=True,
    )
    turn = Turn(
        index=0,
        id="turn-1",
        trigger=UserInput(content=(TextContent(type="text", text="hello from user"),)),
        rounds=(
            Round(
                response=AssistantMessage(
                    role="assistant",
                    content=(
                        TextContent(type="text", text="working"),
                        ThinkingBlock(type="thinking", text="private reasoning"),
                        call,
                    ),
                    timestamp=0.0,
                ),
                tool_results=(ToolRecord(call=call, result=result),),
            ),
        ),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=0.0,
        meta=TurnMeta(
            total_input_tokens=10,
            total_output_tokens=5,
            model_id="test-model",
            system_prompt="system prompt",
        ),
    )
    snapshot = build_trace_snapshot("session-1", [turn])

    views = viewer._build_views(snapshot)

    assert len(views) == 1
    view = views[0]
    assert view.summary is snapshot.turns[0]
    assert view.trigger_label == '"hello from user"'
    assert view.tools_str == "1 tools (1 err)"
    assert view.cause == "ModelEndTurn"
    assert view.in_tok == 10
    assert view.out_tok == 5
    assert view.model == "test-model"
    assert [message.role for message in view.messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool_call",
        "tool_result",
    ]
    assert view.messages[2].content == "working"
    assert view.messages[3].thinking == "private reasoning"
    assert view.messages[4].tool_name == "bash"
    assert '"cmd": "echo hi"' in (view.messages[4].args_json or "")
    assert view.messages[5].is_error is True
    assert view.messages[5].content == "boom"
