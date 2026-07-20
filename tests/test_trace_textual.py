from __future__ import annotations

import pytest
from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text
from textual.coordinate import Coordinate
from textual.widgets import DataTable

from agentm.cli._trace_model import TraceRow
from agentm.cli._trace_textual import (
    TraceConsoleApp,
    TrajectoryDataSource,
    _detail_renderable,
)
from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultBlock,
)
from agentm.core.abi.query import SessionFilter, SessionIdentity
from agentm.core.abi.termination import ModelEndTurn
from agentm.core.abi.trajectory import Outcome, Round, ToolRecord, Turn, TurnCheckpoint
from agentm.core.abi.trigger import UserInput


class _Query:
    def __init__(self, turn: Turn) -> None:
        self._turn = turn

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> list[SessionIdentity]:
        return [SessionIdentity(id="session-1")]

    def turns(self, session_id: str) -> list[Turn]:
        return [self._turn]

    def checkpoints(self, session_id: str) -> list[TurnCheckpoint]:
        return []


def _turn_with_many_rounds(count: int = 6) -> Turn:
    rounds: list[Round] = []
    for index in range(count):
        call = ToolCallBlock(
            type="tool_call",
            id=f"call-{index}",
            name="bash",
            arguments={"cmd": f"echo {index}"},
        )
        result = ToolResultBlock(
            type="tool_result",
            tool_call_id=call.id,
            content=(TextContent(type="text", text=f"result {index}"),),
        )
        rounds.append(
            Round(
                response=AssistantMessage(
                    role="assistant",
                    content=(
                        TextContent(type="text", text=f"assistant {index}"),
                        call,
                    ),
                    timestamp=0.0,
                ),
                tool_results=(ToolRecord(call=call, result=result),),
            )
        )
    return Turn(
        index=0,
        id="turn-1",
        trigger=UserInput(content=(TextContent(type="text", text="hello"),)),
        rounds=tuple(rounds),
        outcome=Outcome(cause=ModelEndTurn()),
        timestamp=0.0,
    )


@pytest.mark.asyncio
async def test_row_highlight_does_not_jump_back_to_turn_start() -> None:
    app = TraceConsoleApp(
        TrajectoryDataSource(_Query(_turn_with_many_rounds()), "session-1")
    )

    async with app.run_test(size=(120, 40)) as pilot:
        rows = app.query_one("#rows", DataTable)
        assert rows.row_count > 6

        rows.move_cursor(row=6, column=0, scroll=True)
        await pilot.pause()

        assert rows.cursor_row == 6


@pytest.mark.asyncio
async def test_follow_refresh_tracks_new_tail_when_cursor_was_at_tail() -> None:
    query = _Query(_turn_with_many_rounds(2))
    app = TraceConsoleApp(TrajectoryDataSource(query, "session-1"), follow=True)

    async with app.run_test(size=(120, 40)) as pilot:
        rows = app.query_one("#rows", DataTable)
        await pilot.pause()
        assert rows.cursor_row == rows.row_count - 1
        assert rows.scroll_y == rows.max_scroll_y

        query._turn = _turn_with_many_rounds(4)
        app._reload_snapshot()
        await pilot.pause()

        assert rows.cursor_row == rows.row_count - 1
        assert rows.scroll_y == rows.max_scroll_y
        assert rows.get_row_at(rows.cursor_row)[0] == "T0"


@pytest.mark.asyncio
async def test_follow_refresh_preserves_history_cursor_when_not_at_tail() -> None:
    query = _Query(_turn_with_many_rounds(3))
    app = TraceConsoleApp(TrajectoryDataSource(query, "session-1"), follow=True)

    async with app.run_test(size=(120, 40)) as pilot:
        rows = app.query_one("#rows", DataTable)
        rows.move_cursor(row=1, column=0, scroll=True)
        await pilot.pause()
        previous_key = rows.coordinate_to_cell_key(Coordinate(1, 0)).row_key.value

        query._turn = _turn_with_many_rounds(5)
        app._reload_snapshot()
        await pilot.pause()

        current_key = rows.coordinate_to_cell_key(rows.cursor_coordinate).row_key.value
        assert rows.cursor_row == 1
        assert current_key == previous_key


@pytest.mark.asyncio
async def test_query_change_to_no_matches_clears_stale_rows() -> None:
    app = TraceConsoleApp(
        TrajectoryDataSource(_Query(_turn_with_many_rounds(2)), "session-1")
    )

    async with app.run_test(size=(120, 40)) as pilot:
        rows = app.query_one("#rows", DataTable)
        assert rows.row_count > 0

        await pilot.press("/")
        await pilot.press("x")
        await pilot.press("y")
        await pilot.press("z")
        await pilot.pause()

        assert rows.row_count == 0


@pytest.mark.asyncio
async def test_turn_pane_width_can_be_dragged() -> None:
    app = TraceConsoleApp(
        TrajectoryDataSource(_Query(_turn_with_many_rounds(2)), "session-1")
    )

    async with app.run_test(size=(140, 40)) as pilot:
        original_width = app._turn_pane_width

        app.begin_turn_pane_resize(40)
        app.drag_turn_pane_resize(52)
        app.end_turn_pane_resize()
        await pilot.pause()

        assert app._turn_pane_width == original_width + 12


@pytest.mark.asyncio
async def test_turn_pane_width_keyboard_resize_is_clamped() -> None:
    app = TraceConsoleApp(
        TrajectoryDataSource(_Query(_turn_with_many_rounds(2)), "session-1")
    )

    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.press("[")
        await pilot.press("[")
        await pilot.press("[")
        await pilot.pause()

        assert app._turn_pane_width >= 24

        await pilot.press("]")
        await pilot.pause()

        assert app._turn_pane_width > 24


def test_tool_call_detail_only_highlights_json_payload() -> None:
    row = TraceRow(
        key="row-1",
        kind="tool_call",
        title="CALL read",
        preview="read",
        content='{\n  "path": "/repo/file.ts",\n  "limit": 160\n}',
        turn_index=0,
        round_index=94,
        status="incomplete",
        tool_name="read",
        metadata={"tool_call_id": "call-1"},
    )

    renderable = _detail_renderable(row)

    assert isinstance(renderable, Group)
    header, spacer, payload = renderable.renderables
    assert isinstance(header, Text)
    assert header.plain.startswith("key: row-1")
    assert "metadata:" in header.plain
    assert isinstance(spacer, Text)
    assert spacer.plain == ""
    assert isinstance(payload, Syntax)
    assert payload.code == row.content
    assert "key:" not in payload.code
