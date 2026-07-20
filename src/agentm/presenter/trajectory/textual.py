# code-health: ignore-file[AM009] -- Textual event handlers stay on the App subclass
"""Textual-powered interactive trajectory app."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

from loguru import logger
from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.geometry import Size
from textual.render import measure
from textual.widgets import DataTable, Footer, Header, Input, Static, Tab, Tabs

from agentm.presenter.trajectory.model import (
    TraceQuery,
    TraceRow,
    TraceSnapshot,
    TraceTableColumn,
    TraceTurnSummary,
    TraceView,
    TraceViewRegistry,
    TraceViewSpec,
    build_trace_snapshot,
    default_trace_view_registry,
    parse_trace_query,
)
from agentm.core.abi.query import TrajectoryQueryStore

_TableRows = tuple[tuple[str, tuple[str, ...], int], ...]

_DEFAULT_TURN_PANE_WIDTH = 32
_MIN_TURN_PANE_WIDTH = 24
_MAX_TURN_PANE_WIDTH = 96
_MIN_MAIN_PANE_WIDTH = 48
_COLUMN_RESIZE_HIT_ZONE = 1
_MIN_COLUMN_CONTENT_WIDTH = 1
_DEFAULT_ROW_COLUMNS = (
    TraceTableColumn("location", "Loc"),
    TraceTableColumn("kind", "Kind"),
    TraceTableColumn("name", "Name"),
    TraceTableColumn("state", "State"),
    TraceTableColumn("preview", "Preview"),
)


@dataclass(frozen=True, slots=True)
class TrajectoryDataSource:
    """Small read-side wrapper used by the Textual app."""

    query: TrajectoryQueryStore
    session_id: str

    def load(self) -> TraceSnapshot:
        turns = list(self.query.turns(self.session_id))
        checkpoints = list(self.query.checkpoints(self.session_id))
        return build_trace_snapshot(self.session_id, turns, checkpoints)


class PaneDivider(Static):
    """Draggable splitter between the turn list and the active trace view."""

    can_focus = False

    def __init__(self) -> None:
        super().__init__("", id="pane-divider")

    def render(self) -> str:
        return "\n".join("|" for _ in range(max(1, self.size.height)))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        event.stop()
        self.capture_mouse()
        self.add_class("dragging")
        app = cast("TraceConsoleApp", self.app)
        app.begin_turn_pane_resize(event.screen_x)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()
        app = cast("TraceConsoleApp", self.app)
        app.drag_turn_pane_resize(event.screen_x)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        event.stop()
        self.release_mouse()
        self.remove_class("dragging")
        app = cast("TraceConsoleApp", self.app)
        app.end_turn_pane_resize()


class ResizableDataTable(DataTable[str]):
    """DataTable variant with content-fit columns and header-edge resizing."""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self._columns_fit_once = False
        self._column_max_widths: tuple[int | None, ...] = ()
        self._column_resize_index: int | None = None
        self._column_resize_start_x = 0
        self._column_resize_start_width = 0

    def on_mouse_down(self, event: events.MouseDown) -> None:
        column_index = self._column_resize_hit_test(event)
        if column_index is None:
            return
        event.stop()
        self.focus()
        self.capture_mouse()
        self.begin_column_resize(column_index, event.screen_x)

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        if self._column_resize_index is None:
            super()._on_mouse_move(event)
            return
        event.stop()
        self.drag_column_resize(event.screen_x)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._column_resize_index is None:
            return
        event.stop()
        self.release_mouse()
        self.end_column_resize()

    async def _on_click(self, event: events.Click) -> None:
        if self._column_resize_hit_test(event) is not None:
            event.stop()
            return
        await super()._on_click(event)

    def begin_column_resize(self, column_index: int, screen_x: int) -> None:
        if not self.is_valid_column_index(column_index):
            return
        column = self.ordered_columns[column_index]
        self._column_resize_index = column_index
        self._column_resize_start_x = screen_x
        self._column_resize_start_width = (
            column.content_width if column.auto_width else column.width
        )

    def drag_column_resize(self, screen_x: int) -> None:
        if self._column_resize_index is None:
            return
        delta = screen_x - self._column_resize_start_x
        self.set_column_width(
            self._column_resize_index,
            self._column_resize_start_width + delta,
        )

    def end_column_resize(self) -> None:
        self._column_resize_index = None

    def reset_columns(self, columns: Sequence[TraceTableColumn]) -> None:
        self.clear(columns=True)
        self._columns_fit_once = False
        self._column_max_widths = tuple(column.max_width for column in columns)
        self._column_resize_index = None
        for column in columns:
            self.add_column(column.title)

    def set_column_width(self, column_index: int, width: int) -> None:
        if not self.is_valid_column_index(column_index):
            return
        self._columns_fit_once = True
        column = self.ordered_columns[column_index]
        previous_render_width = column.get_render_width(self)
        column.auto_width = False
        column.width = max(_MIN_COLUMN_CONTENT_WIDTH, width)
        next_render_width = column.get_render_width(self)
        if next_render_width == previous_render_width:
            return
        self._update_count += 1
        self.virtual_size = Size(
            max(0, self.virtual_size.width + next_render_width - previous_render_width),
            self.virtual_size.height,
        )
        self.refresh()

    def fit_columns_once(self, rows: _TableRows) -> None:
        if self._columns_fit_once or not rows:
            return
        for column_index, column in enumerate(self.ordered_columns):
            width = measure(self.app.console, column.label, 1)
            for _key, cells, _content_hash in rows:
                if column_index < len(cells):
                    width = max(
                        width,
                        measure(
                            self.app.console,
                            Text.from_markup(cells[column_index], end=""),
                            1,
                        ),
                    )
            if column_index < len(self._column_max_widths):
                max_width = self._column_max_widths[column_index]
                if max_width is not None:
                    width = min(width, max_width)
            self.set_column_width(column_index, width)
        self._columns_fit_once = True

    def _column_resize_hit_test(self, event: events.MouseEvent) -> int | None:
        meta = event.style.meta
        if meta.get("row") != -1 or not self.show_header:
            return None
        right_edge = self._row_label_column_width - self.scroll_offset.x
        for column_index, column in enumerate(self.ordered_columns):
            right_edge += column.get_render_width(self)
            if abs(event.x - (right_edge - 1)) <= _COLUMN_RESIZE_HIT_ZONE:
                return column_index
        return None


class TraceConsoleApp(App[None]):
    """Interactive trajectory explorer with pluggable views."""

    CSS = """
    Screen {
        background: $surface;
    }

    #shell {
        height: 1fr;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #body {
        height: 1fr;
    }

    #turn-pane {
        width: 32;
        min-width: 24;
        height: 1fr;
        border: solid $primary;
    }

    #pane-divider {
        width: 1;
        min-width: 1;
        height: 1fr;
        background: $primary;
        color: $text;
        content-align: center middle;
    }

    #pane-divider:hover,
    #pane-divider.dragging {
        background: $accent;
    }

    #main-pane {
        width: 1fr;
        height: 1fr;
    }

    #turn-title,
    #detail-title {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text;
    }

    #view-tabs {
        height: 3;
    }

    #query {
        height: 3;
    }

    #rows {
        height: 2fr;
    }

    #detail-scroll {
        height: 1fr;
        border: solid $accent;
    }

    #detail {
        width: 1fr;
        padding: 0 1;
    }

    DataTable {
        height: 1fr;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("/", "focus_query", "Search"),
        Binding("escape", "clear_focus", "Blur"),
        Binding("tab", "next_view", "Next view"),
        Binding("shift+tab", "previous_view", "Previous view"),
        Binding("1", "view_index(0)", "Trajectory", show=False),
        Binding("2", "view_index(1)", "Tools", show=False),
        Binding("3", "view_index(2)", "Errors", show=False),
        Binding("4", "view_index(3)", "Metrics", show=False),
        Binding("5", "view_index(4)", "Policy", show=False),
        Binding("6", "view_index(5)", "View 6", show=False),
        Binding("7", "view_index(6)", "View 7", show=False),
        Binding("8", "view_index(7)", "View 8", show=False),
        Binding("9", "view_index(8)", "View 9", show=False),
        Binding("0", "view_index(9)", "View 10", show=False),
        Binding("[", "resize_turn_pane(-4)", "Narrow", show=False),
        Binding("]", "resize_turn_pane(4)", "Widen", show=False),
        Binding("n", "next_match", "Next"),
        Binding("N", "previous_match", "Prev"),
    ]

    def __init__(
        self,
        data_source: TrajectoryDataSource,
        *,
        follow: bool = False,
        registry: TraceViewRegistry | None = None,
    ) -> None:
        super().__init__()
        self._data_source = data_source
        self._follow = follow
        self._view_registry = registry or default_trace_view_registry()
        self._specs = self._view_registry.specs()
        self._active_view_id = self._specs[0].id if self._specs else "trajectory"
        self._query = TraceQuery()
        self._snapshot: TraceSnapshot | None = None
        self._view: TraceView | None = None
        self._rows_by_key: dict[str, TraceRow] = {}
        self._turns_by_key: dict[str, TraceTurnSummary] = {}
        self._row_keys: list[str] = []
        self._turn_table_rows: _TableRows = ()
        self._row_table_rows: _TableRows = ()
        self._row_columns: tuple[TraceTableColumn, ...] = ()
        self._turn_pane_width = _DEFAULT_TURN_PANE_WIDTH
        self._resize_start_x: int | None = None
        self._resize_start_width = self._turn_pane_width

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="shell"):
            yield Static("", id="status")
            with Horizontal(id="body"):
                with Vertical(id="turn-pane"):
                    yield Static("Turns", id="turn-title")
                    yield ResizableDataTable(id="turns")
                yield PaneDivider()
                with Vertical(id="main-pane"):
                    yield Tabs(
                        *(Tab(spec.title, id=spec.id) for spec in self._specs),
                        id="view-tabs",
                    )
                    yield Input(
                        placeholder=(
                            "Search/filter: text, error, role:assistant, "
                            "tool:bash, status:incomplete, cause:ModelEndTurn"
                        ),
                        id="query",
                    )
                    yield ResizableDataTable(id="rows")
                    yield Static("Detail", id="detail-title")
                    with VerticalScroll(id="detail-scroll"):
                        yield Static("", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "AgentM Trace"
        self.sub_title = self._data_source.session_id
        self._configure_tables()
        self._apply_turn_pane_width()
        self._reload_snapshot()
        if self._follow:
            self.set_interval(1.5, self._reload_snapshot)

    def _configure_tables(self) -> None:
        turns = self.query_one("#turns", DataTable)
        turns.cursor_type = "row"
        turns.zebra_stripes = True
        turns.add_column("T")
        turns.add_column("State")
        turns.add_column("R")
        turns.add_column("Tools")

        rows = self.query_one("#rows", DataTable)
        rows.cursor_type = "row"
        rows.zebra_stripes = True
        self._configure_row_columns(_DEFAULT_ROW_COLUMNS)

    def _configure_row_columns(
        self,
        columns: tuple[TraceTableColumn, ...],
    ) -> None:
        if columns == self._row_columns:
            return
        table = self.query_one("#rows", ResizableDataTable)
        table.reset_columns(columns)
        self._row_columns = columns
        self._row_table_rows = ()

    def _reload_snapshot(self) -> None:
        try:
            self._snapshot = self._data_source.load()
        except KeyError:
            self.query_one("#status", Static).update(
                f"Session not found: {self._data_source.session_id}"
            )
            return
        self._render_current_state(from_refresh=True)

    def _render_current_state(self, *, from_refresh: bool = False) -> None:
        if self._snapshot is None:
            return
        spec = self._active_spec()
        self._view = spec.build(self._snapshot, self._query)
        self._render_status(spec)
        self._render_turns(follow_tail=from_refresh and self._follow)
        self._render_rows(follow_tail=from_refresh and self._follow)
        self._render_detail_for_cursor()

    def _render_status(self, spec: TraceViewSpec) -> None:
        if self._snapshot is None or self._view is None:
            return
        query = self._query.raw.strip() or "-"
        follow = "follow:on" if self._follow else "follow:off"
        self.query_one("#status", Static).update(
            (
                f"sid:{_short_id(self._snapshot.session_id)} | "
                f"{spec.id} | {self._snapshot.status_label} | "
                f"rows:{len(self._view.rows)} | "
                f"turns:{self._snapshot.metrics.committed_turns}"
                f"+{self._snapshot.metrics.incomplete_turns} | "
                f"err:{self._snapshot.metrics.tool_errors} | "
                f"query:{query} | {follow}"
            )
        )

    def _render_turns(self, *, follow_tail: bool = False) -> None:
        if self._snapshot is None:
            return
        table = self.query_one("#turns", ResizableDataTable)
        previous_key = self._current_row_key(table)
        should_follow_tail = follow_tail and self._is_table_at_tail(table)
        self._turns_by_key = {}
        next_rows: list[tuple[str, tuple[str, ...], int]] = []
        for summary in self._snapshot.turns:
            self._turns_by_key[summary.key] = summary
            tools = f"{summary.tool_calls}"
            if summary.tool_errors:
                tools += f"/{summary.tool_errors}e"
            cells = (
                str(summary.turn_index),
                summary.state_label,
                str(summary.rounds),
                tools,
            )
            next_rows.append((summary.key, cells, 0))
        self._turn_table_rows = self._sync_table_rows(
            table,
            previous=self._turn_table_rows,
            next_rows=tuple(next_rows),
        )
        self._restore_cursor(table, previous_key, follow_tail=should_follow_tail)

    def _render_rows(self, *, follow_tail: bool = False) -> None:
        if self._view is None:
            return
        table = self.query_one("#rows", ResizableDataTable)
        previous_key = self._current_row_key(table)
        self._configure_row_columns(self._view.columns or _DEFAULT_ROW_COLUMNS)
        should_follow_tail = follow_tail and self._is_table_at_tail(table)
        self._rows_by_key = {}
        self._row_keys = []
        next_rows: list[tuple[str, tuple[str, ...], int]] = []
        for row in self._view.rows:
            self._rows_by_key[row.key] = row
            self._row_keys.append(row.key)
            cells = tuple(
                _row_column_value(row, column.key) for column in self._row_columns
            )
            next_rows.append((row.key, cells, hash(row.content)))
        self._row_table_rows = self._sync_table_rows(
            table,
            previous=self._row_table_rows,
            next_rows=tuple(next_rows),
        )
        self._restore_cursor(table, previous_key, follow_tail=should_follow_tail)
        if not self._row_keys:
            self.query_one("#detail", Static).update(self._view.empty_text)

    def _render_detail_for_cursor(self) -> None:
        table = self.query_one("#rows", DataTable)
        row_key = self._current_row_key(table)
        row = self._rows_by_key.get(row_key or "")
        if row is None:
            if self._view is not None:
                self.query_one("#detail", Static).update(self._view.empty_text)
            return
        self._render_detail(row)

    def _render_detail(self, row: TraceRow) -> None:
        title = row.title
        location = row.location
        tool = f" | tool:{row.tool_name}" if row.tool_name else ""
        state = row.cause or row.status or "-"
        self.query_one("#detail-title", Static).update(
            f"{location} | {row.kind} | {state}{tool} | {title}"
        )
        detail = self.query_one("#detail", Static)
        detail.update(_detail_renderable(row))

    def _active_spec(self) -> TraceViewSpec:
        for spec in self._specs:
            if spec.id == self._active_view_id:
                return spec
        return self._specs[0]

    @on(Tabs.TabActivated, "#view-tabs")
    def _on_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab.id is None:
            return
        self._active_view_id = str(event.tab.id)
        self._render_current_state()

    @on(Input.Changed, "#query")
    def _on_query_changed(self, event: Input.Changed) -> None:
        self._query = parse_trace_query(event.value)
        self._row_table_rows = ()
        self._render_current_state()

    @on(Input.Submitted, "#query")
    def _on_query_submitted(self, event: Input.Submitted) -> None:
        self._query = parse_trace_query(event.value)
        self._row_table_rows = ()
        self.query_one("#rows", DataTable).focus()
        self._render_current_state()

    @on(DataTable.RowHighlighted, "#rows")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value
        if key is None:
            return
        row = self._rows_by_key.get(key)
        if row is not None:
            self._render_detail(row)
            self._select_turn(row.turn_index)

    @on(DataTable.RowSelected, "#turns")
    def _on_turn_selected(self, event: DataTable.RowSelected) -> None:
        key = event.row_key.value
        if key is None:
            return
        summary = self._turns_by_key.get(key)
        if summary is not None:
            self._jump_to_turn(summary.turn_index)

    def action_refresh(self) -> None:
        self._reload_snapshot()

    def action_focus_query(self) -> None:
        self.query_one("#query", Input).focus()

    def action_clear_focus(self) -> None:
        query = self.query_one("#query", Input)
        if self.focused is query:
            self.query_one("#rows", DataTable).focus()
        else:
            query.value = ""

    def action_next_view(self) -> None:
        self._activate_view_offset(1)

    def action_previous_view(self) -> None:
        self._activate_view_offset(-1)

    def action_view_index(self, index: int) -> None:
        if 0 <= index < len(self._specs):
            self._activate_view_id(self._specs[index].id)

    def action_next_match(self) -> None:
        table = self.query_one("#rows", DataTable)
        if self._row_keys:
            table.action_cursor_down()

    def action_previous_match(self) -> None:
        table = self.query_one("#rows", DataTable)
        if self._row_keys:
            table.action_cursor_up()

    def action_resize_turn_pane(self, delta: int) -> None:
        self.set_turn_pane_width(self._turn_pane_width + delta)

    def begin_turn_pane_resize(self, screen_x: int) -> None:
        self._resize_start_x = screen_x
        self._resize_start_width = self._turn_pane_width

    def drag_turn_pane_resize(self, screen_x: int) -> None:
        if self._resize_start_x is None:
            return
        delta = screen_x - self._resize_start_x
        self.set_turn_pane_width(self._resize_start_width + delta)

    def end_turn_pane_resize(self) -> None:
        self._resize_start_x = None
        self._resize_start_width = self._turn_pane_width

    def set_turn_pane_width(self, width: int) -> None:
        clamped = self._clamp_turn_pane_width(width)
        if clamped == self._turn_pane_width:
            return
        self._turn_pane_width = clamped
        self._apply_turn_pane_width()

    def _apply_turn_pane_width(self) -> None:
        pane = self.query_one("#turn-pane", Vertical)
        pane.styles.width = self._turn_pane_width

    def _clamp_turn_pane_width(self, width: int) -> int:
        available = self.size.width
        max_for_terminal = (
            available - _MIN_MAIN_PANE_WIDTH - 1 if available > 0 else width
        )
        maximum = max(_MIN_TURN_PANE_WIDTH, min(_MAX_TURN_PANE_WIDTH, max_for_terminal))
        return max(_MIN_TURN_PANE_WIDTH, min(maximum, width))

    def _activate_view_offset(self, delta: int) -> None:
        if not self._specs:
            return
        index = next(
            (
                i
                for i, spec in enumerate(self._specs)
                if spec.id == self._active_view_id
            ),
            0,
        )
        self._activate_view_id(self._specs[(index + delta) % len(self._specs)].id)

    def _activate_view_id(self, view_id: str) -> None:
        self._active_view_id = view_id
        self._row_table_rows = ()
        tabs = self.query_one("#view-tabs", Tabs)
        tabs.active = view_id
        self._render_current_state()

    def _jump_to_turn(self, turn_index: int) -> None:
        table = self.query_one("#rows", DataTable)
        for index, row in enumerate(self._view.rows if self._view is not None else ()):
            if row.turn_index == turn_index:
                table.move_cursor(row=index, column=0, scroll=True)
                return

    def _select_turn(self, turn_index: int | None) -> None:
        if turn_index is None or self._snapshot is None:
            return
        table = self.query_one("#turns", DataTable)
        for index, summary in enumerate(self._snapshot.turns):
            if summary.turn_index == turn_index:
                table.move_cursor(row=index, column=0, scroll=True)
                return

    def _current_row_key(self, table: DataTable) -> str | None:
        if table.row_count == 0:
            return None
        try:
            value = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception as exc:
            logger.debug("trace textual row-key lookup failed: {}", exc)
            return None
        return cast(str | None, value)

    def _restore_cursor(
        self,
        table: ResizableDataTable,
        previous_key: str | None,
        *,
        follow_tail: bool = False,
    ) -> None:
        if table.row_count == 0:
            return
        if follow_tail:
            self._move_table_to_tail(table)
            return
        if previous_key is not None:
            for row_index in range(table.row_count):
                key = table.coordinate_to_cell_key(
                    Coordinate(row_index, 0)
                ).row_key.value
                if key == previous_key:
                    table.move_cursor(row=row_index, column=0, scroll=False)
                    return
        table.move_cursor(row=0, column=0, scroll=False)

    def _move_table_to_tail(self, table: DataTable) -> None:
        table.move_cursor(row=table.row_count - 1, column=0, scroll=True)
        table.scroll_end(animate=False, immediate=True, x_axis=False)

        def scroll_after_layout() -> None:
            if table.row_count:
                table.scroll_end(animate=False, immediate=True, x_axis=False)

        self.call_after_refresh(scroll_after_layout)

    def _is_table_at_tail(self, table: DataTable) -> bool:
        return table.row_count == 0 or table.cursor_row >= table.row_count - 1

    def _sync_table_rows(
        self,
        table: ResizableDataTable,
        *,
        previous: _TableRows,
        next_rows: _TableRows,
    ) -> _TableRows:
        table_matches_previous = table.row_count == len(previous)
        if previous == next_rows and table_matches_previous:
            return previous
        if not previous and table.row_count == 0:
            table.fit_columns_once(next_rows)

        previous_keys = tuple(row[0] for row in previous)
        next_keys = tuple(row[0] for row in next_rows)
        can_patch = (
            table_matches_previous
            and len(next_rows) >= len(previous)
            and next_keys[: len(previous_keys)] == previous_keys
        )
        if not can_patch:
            table.clear()
            table.fit_columns_once(next_rows)
            for key, cells, _ in next_rows:
                table.add_row(*cells, key=key)
            return next_rows

        for row_index, (key, cells, _content_hash) in enumerate(next_rows):
            if row_index >= len(previous):
                table.add_row(*cells, key=key)
                continue
            _previous_key, previous_cells, _previous_hash = previous[row_index]
            for column_index, cell in enumerate(cells):
                if (
                    column_index >= len(previous_cells)
                    or previous_cells[column_index] != cell
                ):
                    table.update_cell_at(Coordinate(row_index, column_index), cell)
        return next_rows


def run_textual_viewer(
    query: TrajectoryQueryStore,
    session_id: str,
    *,
    follow: bool = False,
    registry: TraceViewRegistry | None = None,
) -> None:
    """Run the Textual trajectory console."""

    app = TraceConsoleApp(
        TrajectoryDataSource(query=query, session_id=session_id),
        follow=follow,
        registry=registry,
    )
    app.run()


def _row_column_value(row: TraceRow, key: str) -> str:
    if key == "location":
        return row.location
    if key == "kind":
        return row.kind
    if key == "name":
        return row.display_name or row.tool_name or row.title
    if key == "state":
        return row.cause or row.status or "-"
    if key == "preview":
        return row.preview
    if key == "tool":
        return row.tool_name or _metadata_text(row.metadata.get(key))
    if key == "title":
        return row.title
    return _metadata_text(row.metadata.get(key))


def _metadata_text(value: object) -> str:
    if value is None or value == "":
        return "-"
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def _detail_renderable(row: TraceRow, *, max_chars: int = 120_000) -> RenderableType:
    header = Text(_detail_header(row), no_wrap=False)
    body = _truncated_content(row.content, max_chars=max_chars)
    if row.kind == "tool_call":
        return Group(
            header,
            Text(""),
            Syntax(body, "json", word_wrap=True, theme="monokai"),
        )
    return Text(f"{header.plain}\n\n{body}", no_wrap=False)


def _detail_content(row: TraceRow, *, max_chars: int = 120_000) -> str:
    return f"{_detail_header(row)}\n\n{_truncated_content(row.content, max_chars=max_chars)}"


def _detail_header(row: TraceRow) -> str:
    header = [
        f"key: {row.key}",
        f"location: {row.location}",
        f"kind: {row.kind}",
        f"status: {row.status or '-'}",
        f"cause: {row.cause or '-'}",
    ]
    if row.tool_name:
        header.append(f"tool: {row.tool_name}")
    if row.metadata:
        header.append("metadata:")
        for key, value in row.metadata.items():
            header.append(f"  {key}: {value}")
    return "\n".join(header)


def _truncated_content(content: str, *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + f"\n\n... truncated {len(content) - max_chars:,} chars ..."
    )


def _short_id(value: str, *, keep: int = 8) -> str:
    if len(value) <= keep * 2 + 1:
        return value
    return f"{value[:keep]}..{value[-keep:]}"


__all__ = [
    "ResizableDataTable",
    "TraceConsoleApp",
    "TrajectoryDataSource",
    "run_textual_viewer",
]
