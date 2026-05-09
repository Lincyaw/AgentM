from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from rich.console import Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table as RichTable
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from agentm.core.abi import (
    AssistantMessage,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    MessageEnd,
    StreamDeltaEvent,
    TextContent,
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    ToolCallEvent,
    ToolCallStart,
    ToolResultEvent,
    Usage,
)
from agentm.harness import AgentSession, AgentSessionConfig
from agentm.harness.events import (
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    CostBudgetExceededEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
)
from agentm.harness.extension import CommandSpec


_QUIT_COMMANDS = frozenset({"/quit", "/exit", "/q", "exit", "quit"})
_BUILTIN_COMMANDS: dict[str, str] = {
    "/quit": "Quit the TUI.",
    "/exit": "Quit the TUI.",
    "/q": "Quit the TUI.",
    "/clear": "Clear the visible conversation log.",
    "/help": "Show key bindings and slash commands.",
    "/copy-last": "Copy the most recent assistant text block.",
    "/extensions": "List loaded extensions and their status.",
    "/tools": "List tools registered with the agent.",
    "/budget": "Show current cost-budget state.",
}

# Reload triggers that represent self-modification — the agent (or an
# approved propose_change) edited an atom under its own catalog. The TUI
# highlights these so the framework's signature behavior is *visible*
# rather than silently happening through git.
_SELF_MODIFY_TRIGGERS = frozenset({"agent", "propose_change_approved"})

_PHASE_GLYPHS: dict[str, str] = {
    "idle": "●",
    "thinking": "◐",
    "streaming": "◑",
    "tool": "▶",
    "subagent": "↳",
}

_THEME_ALIASES: dict[str, str] = {
    "dark": "textual-dark",
    "light": "textual-light",
}


class _SessionLike(Protocol):
    @property
    def bus(self) -> Any: ...

    async def prompt(
        self,
        text: str,
        *,
        images: list[Any] | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[Any]: ...

    async def shutdown(self) -> None: ...

    @property
    def model(self) -> Any: ...


@dataclass(slots=True)
class SlashCommandEntry:
    name: str
    description: str
    source: str


@dataclass(slots=True)
class ToolEntry:
    name: str
    description: str
    source: str


@dataclass(slots=True)
class ToolRenderState:
    tool_name: str
    start_ns: int
    args: dict[str, Any] | None = None
    args_json_fragments: list[str] | None = None


class ConversationLog(VerticalScroll):
    pass


class StatusHeader(Static):
    """Top dock-bar showing model · turn · tokens · cost · phase.

    Rendered as a single-line strip at the top of the app. Replaces the
    legacy bottom italic-dim status row; that one was unreadable and the
    information density warrants prime real estate.
    """

    model_name: reactive[str] = reactive("?")
    current_text: reactive[str] = reactive(
        "AgentM  ▎?  ·  turn 0  ·  in 0  out 0  ·  $0.000  ·  ● idle"
    )
    turn_number: reactive[int] = reactive(0)
    tokens_in: reactive[int] = reactive(0)
    tokens_out: reactive[int] = reactive(0)
    cost_usd: reactive[float] = reactive(0.0)
    phase: reactive[str] = reactive("idle")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Plain (non-reactive) shadow state for registry counters and
        # budget. They are derived from ``AgentMApp`` dicts, so giving
        # them their own reactive setters would just create a second
        # source of truth that has to be kept in sync. ``set_registry``
        # / ``set_budget_exceeded`` are the single mutation entry points.
        self._extensions_loaded = 0
        self._extensions_failed = 0
        self._tools_registered = 0
        self._budget_exceeded = False

    def watch_model_name(self, _: str) -> None:
        self._refresh()

    def watch_turn_number(self, _: int) -> None:
        self._refresh()

    def watch_tokens_in(self, _: int) -> None:
        self._refresh()

    def watch_tokens_out(self, _: int) -> None:
        self._refresh()

    def watch_cost_usd(self, _: float) -> None:
        self._refresh()

    def watch_phase(self, _: str) -> None:
        self._refresh()

    def set_registry(self, *, loaded: int, failed: int, tools: int) -> None:
        if (loaded, failed, tools) == (
            self._extensions_loaded,
            self._extensions_failed,
            self._tools_registered,
        ):
            return
        self._extensions_loaded = loaded
        self._extensions_failed = failed
        self._tools_registered = tools
        self._refresh()

    def set_budget_exceeded(self, value: bool) -> None:
        if value == self._budget_exceeded:
            return
        self._budget_exceeded = value
        self._refresh()

    def _refresh(self) -> None:
        glyph = _PHASE_GLYPHS.get(self.phase, "●")
        ext_segment = f"{self._extensions_loaded} ext"
        if self._extensions_failed:
            ext_segment = f"{ext_segment} ({self._extensions_failed} failed)"
        cost_segment = f"${self.cost_usd:.3f}"
        if self._budget_exceeded:
            cost_segment = f"{cost_segment} ⚠"
        self.current_text = (
            f"AgentM  ▎{self.model_name}"
            f"  ·  turn {self.turn_number}"
            f"  ·  {ext_segment}"
            f"  ·  {self._tools_registered} tools"
            f"  ·  in {self.tokens_in}  out {self.tokens_out}"
            f"  ·  {cost_segment}"
            f"  ·  {glyph} {self.phase}"
        )
        self.update(self.current_text)


class AssistantTextBlock(Static):
    def __init__(self) -> None:
        super().__init__(id="assistant-text")
        self.text = ""
        self.update(Markdown(""))

    def set_text(self, text: str) -> None:
        self.text = text
        self.update(Markdown(text or ""))


class ThinkingBlock(Static):
    """Inline italic-dim thinking preview, no Collapsible chrome.

    The previous design wrapped the thinking buffer in a ``Collapsible``
    so the user could fold it. Claude Code renders thinking as a quiet
    italic-dim block inline before the response and just hides it once
    the response is complete — that's what we do here. ``set_text``
    drives both visibility and body content; ``collapse()`` hides the
    block (called from ``handle_llm_end``) without losing ``self.text``.
    """

    def __init__(self) -> None:
        super().__init__(classes="thinking")
        self.text = ""
        self.display = False

    def set_text(self, text: str) -> None:
        self.text = text
        stripped = text.strip()
        if stripped:
            self.display = True
            self.update(Text(stripped, style="italic"))
        else:
            self.display = False
            self.update("")

    def collapse(self) -> None:
        """Hide the block on response complete; ``self.text`` is preserved
        so an external observer can still read what the model thought
        about."""
        self.display = False


class ToolCallBlock(Collapsible):
    """Tool call rendered as a Collapsible — Claude-Code-style.

    Title format: ``tool_name(args_summary)  ✓ 142ms``. ``args_summary``
    is a short inline preview (single-arg → just the value; multi-arg →
    ``key=val, key=val``) so even when collapsed the user sees what was
    called. The body is indented 4 spaces (CSS ``.tool-body``) and its
    sections are introduced by ``⎿  args`` / ``⎿  result`` connectors,
    mirroring the tree-style Claude Code uses.
    """

    can_focus = True

    def __init__(self, tool_call_id: str, tool_name: str) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.body = Static(classes="tool-body")
        super().__init__(self.body, title=f"{tool_name}(…)", collapsed=False)
        self.add_class("tool-call")
        self.result_text = ""
        self.args: dict[str, Any] = {}
        self.body_kind = "text"
        self.ok: bool | None = None
        self.duration_ms = 0

    def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed

    def set_pending_args(self, args: dict[str, Any], args_preview: str) -> None:
        del args_preview  # legacy param; title now derives from args directly
        self.args = dict(args)
        self.title = f"{self.tool_name}({_format_args_inline(self.args)})"
        self._render_body(None)

    def set_result(self, *, result: Any, duration_ms: int) -> None:
        self.duration_ms = duration_ms
        self.ok = not getattr(result, "is_error", False)
        self.result_text = _tool_result_text(result)
        status = "✓" if self.ok else "✗"
        self.title = (
            f"{self.tool_name}({_format_args_inline(self.args)})"
            f"  {status} {duration_ms}ms"
        )
        self.collapsed = self.ok and _line_count(self.result_text) < 20
        self._render_body(result)
        if self.ok:
            self.remove_class("failed")
        else:
            self.add_class("failed")

    def _render_body(self, result: Any | None) -> None:
        args_text = _format_full_args(self.args) if self.args else "{}"
        if result is None:
            renderable: Any = Group(
                Text("⎿  args", style="dim"),
                Text(args_text),
            )
            self.body_kind = "text"
        else:
            result_renderable = _render_tool_result(self.tool_name, self.result_text)
            self.body_kind = type(result_renderable).__name__
            ok = not getattr(result, "is_error", False)
            renderable = Group(
                Text("⎿  args", style="dim"),
                Text(args_text),
                Text(""),
                Text("⎿  result", style="dim green" if ok else "dim red"),
                result_renderable,
            )
        self.body.update(renderable)


class SubagentBlock(Vertical):
    """Subagent panel — Claude-Code-style ``● subagent: <purpose>`` title
    plus indented body. The body is left-padded by CSS so the lines sit
    visually under the title's argument area, the same way ``ToolCallBlock``
    indents its result. No left gutter / panel border; the visual cue
    is the bullet + indent, nothing else.
    """

    def __init__(self, child_session_id: str, purpose: str) -> None:
        super().__init__(classes="subagent")
        self.child_session_id = child_session_id
        self.purpose = purpose
        self.lines: list[str] = []
        self.failed = False
        self.title_widget = Static(f"● subagent: {purpose}", classes="subagent-title")
        self.body_widget = Static("", classes="subagent-body")

    def compose(self) -> ComposeResult:
        yield self.title_widget
        yield self.body_widget

    def add_line(self, text: str) -> None:
        if not text:
            return
        self.lines.append(text)
        self._refresh_body()

    def mark_error(self, error: str | None) -> None:
        self.failed = bool(error)
        if error:
            self.lines.append(f"error: {error}")
        self.set_class(self.failed, "failed")
        self._refresh_body()

    def _refresh_body(self) -> None:
        if not self.lines:
            return
        if self.is_mounted:
            self.body_widget.update(Markdown("\n\n".join(self.lines)))


class UserTurn(Static):
    """Visual block for a user message — Claude-Code-style ``> `` prefix.

    Each line of the user input gets a bold ``> `` marker; multi-line
    input keeps the marker per line so block quoting reads naturally.
    No widget chrome — color comes from the ``.user-turn`` CSS class
    so theme switches flow through.

    ``injected_from`` distinguishes a real keystroke-from-user message
    (None) from a synthetic one routed through
    ``api.send_user_message`` from inside an extension. Injected turns
    pick up ``.user-turn.injected`` (warning color) plus a small
    ``↪ from <ext>`` trailer so the user can tell at a glance that
    this turn was synthesized rather than typed.
    """

    def __init__(self, text: str, *, injected_from: str | None = None) -> None:
        css_class = "user-turn injected" if injected_from else "user-turn"
        super().__init__(classes=css_class)
        lines = text.splitlines() or [text]
        rendered = Text()
        for i, line in enumerate(lines):
            if i:
                rendered.append("\n")
            rendered.append("> ", style="bold")
            rendered.append(line)
        if injected_from:
            rendered.append(f"\n  ↪ from {injected_from}", style="italic")
        self.update(rendered)


class TurnContainer(Vertical):
    """Container for one assistant turn — thinking + assistant text +
    tool calls + subagent blocks. No leading ``● assistant`` label;
    the chat reads as a flow where assistant content (no prefix) is
    the default and user input is marked with ``> ``. Thinking sits
    above the response in italic-dim, matching Claude Code's layout.
    """

    def __init__(self, logical_turn: int) -> None:
        super().__init__(classes="turn")
        self.logical_turn = logical_turn
        self.thinking = ThinkingBlock()
        self.assistant = AssistantTextBlock()
        self.tools: dict[str, ToolCallBlock] = {}
        self.subagents: dict[str, SubagentBlock] = {}
        self.text_buffer = ""
        self.thinking_buffer = ""

    def compose(self) -> ComposeResult:
        yield self.thinking
        yield self.assistant

    async def ensure_tool(self, tool_call_id: str, tool_name: str) -> ToolCallBlock:
        block = self.tools.get(tool_call_id)
        if block is None:
            block = ToolCallBlock(tool_call_id=tool_call_id, tool_name=tool_name)
            self.tools[tool_call_id] = block
            await self.mount(block)
        return block

    async def add_subagent(self, child_session_id: str, purpose: str) -> SubagentBlock:
        block = SubagentBlock(child_session_id=child_session_id, purpose=purpose)
        self.subagents[child_session_id] = block
        await self.mount(block)
        return block


class PromptInput(TextArea):
    def __init__(self) -> None:
        super().__init__(id="prompt-input")
        self.show_line_numbers = False
        self.soft_wrap = True
        # Tab moves focus rather than indenting — App-level Tab binding
        # was previously misleading because TextArea ate the key when
        # focused. With "focus" here, tab leaves the input naturally.
        self.tab_behavior = "focus"

    async def on_key(self, event: events.Key) -> None:
        app = self.app
        if not isinstance(app, AgentMApp):
            return
        key = event.key
        if key == "enter":
            event.prevent_default()
            event.stop()
            await app.submit_input()
            return
        if key == "shift+enter":
            self.insert("\n")
            event.prevent_default()
            event.stop()
            return
        if key == "escape":
            event.prevent_default()
            event.stop()
            await app.action_soft_cancel()
            return
        if key in {"up", "down"} and not self.text.strip():
            handled = app.cycle_history(-1 if key == "up" else 1)
            if handled:
                event.prevent_default()
                event.stop()
            return
        if key in {"pageup", "pagedown"}:
            # TextArea would otherwise consume these to navigate within
            # the (1-8 line) prompt, which is meaningless and traps the
            # user — they expect PgUp/PgDn to scroll the conversation
            # log. Forward to the app-level scroll actions.
            event.prevent_default()
            event.stop()
            if key == "pageup":
                app.action_scroll_page_up()
            else:
                app.action_scroll_page_down()
            return
        if event.character == "/" and not self.text and self.cursor_location == (0, 0):
            event.prevent_default()
            event.stop()
            app.open_command_palette(initial="")
            return
        app.refresh_input_height()


_ModalResultT = TypeVar("_ModalResultT")


class _DismissibleModal(ModalScreen[_ModalResultT | None], Generic[_ModalResultT]):
    """Modal screen base that closes on ``Esc``.

    The cancel action is identical for all three modals we ship; pulling
    it up here removes the ``ScreenStackError`` guard that was
    triplicated. The guard exists because ``Esc`` can fire after the
    modal has already begun popping (race against another close path).
    """

    BINDINGS = [Binding("escape", "cancel", "Close")]

    def action_cancel(self) -> None:
        if self.app.screen is not self:
            return
        try:
            self.dismiss(None)
        except ScreenStackError:
            return


class CommandPaletteScreen(_DismissibleModal[str]):
    def __init__(self, commands: list[SlashCommandEntry], *, initial: str = "") -> None:
        super().__init__()
        self._commands = commands
        self._initial = initial
        self._filtered = list(commands)

    def compose(self) -> ComposeResult:
        with Container(id="command-palette"):
            yield Input(
                value=self._initial, placeholder="Filter commands", id="command-filter"
            )
            yield OptionList(id="command-options")

    def on_mount(self) -> None:
        self._refresh_options(self._initial)
        self.query_one("#command-filter", Input).focus()

    @on(Input.Changed, "#command-filter")
    def _on_changed(self, event: Input.Changed) -> None:
        self._refresh_options(event.value)

    @on(Input.Submitted, "#command-filter")
    def _on_submitted(self, _: Input.Submitted) -> None:
        if self.app.screen is not self:
            return
        options = self.query_one("#command-options", OptionList)
        try:
            if options.option_count == 0 or options.highlighted is None:
                self.dismiss(None)
                return
            option = options.get_option_at_index(options.highlighted)
            self.dismiss(option.id or str(option.prompt))
        except ScreenStackError:
            return

    @on(OptionList.OptionSelected, "#command-options")
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.app.screen is not self:
            return
        try:
            self.dismiss(event.option.id or str(event.option.prompt))
        except ScreenStackError:
            return

    def _refresh_options(self, query: str) -> None:
        needle = query.strip().lower()
        if needle:
            self._filtered = [
                command
                for command in self._commands
                if needle in command.name.lower()
                or needle in command.description.lower()
            ]
        else:
            self._filtered = list(self._commands)
        options = self.query_one("#command-options", OptionList)
        options.clear_options()
        for command in self._filtered:
            options.add_option(
                Option(f"{command.name} — {command.description}", id=command.name)
            )
        if options.option_count:
            options.highlighted = 0


class HelpScreen(_DismissibleModal[None]):
    def __init__(self, commands: list[SlashCommandEntry]) -> None:
        super().__init__()
        command_lines = "\n".join(
            f"- `{entry.name}` — {entry.description}" for entry in commands
        )
        self._markdown = Markdown(
            "# Help\n\n"
            "## Keys\n"
            "- `Enter` submit\n"
            "- `Shift+Enter` newline\n"
            "- `Esc` soft-cancel (in-flight prompt) / clear draft / no-op (toast)\n"
            "- `Ctrl+C` interrupt; press twice within 1.5s to quit\n"
            "- `Ctrl+D` quit immediately\n"
            "- `Ctrl+L` clear visible log (session memory preserved)\n"
            "- `Ctrl+R` open command palette\n"
            "- `Tab` move focus (input ↔ tool blocks ↔ log)\n"
            "- `PageUp` / `PageDown` scroll log\n"
            "- `Up` / `Down` history when input is empty\n"
            "- `Ctrl+E` toggle focused tool block\n\n"
            "## Control / observability\n"
            "- `/extensions` list loaded extensions and their status\n"
            "- `/tools` list every tool the agent can call right now\n"
            "- `/budget` show current cost-budget state (exceeded?)\n"
            "- Header shows live: `N ext (M failed) · K tools · $cost ⚠`\n"
            "- Self-modification reloads (`agent` / `propose_change_approved`) "
            "raise a `★ self-modify` warning toast\n"
            "- Extension-injected user messages render with a yellow gutter "
            "and a `system → you` label\n\n"
            "## Slash commands\n"
            f"{command_lines}"
        )

    def compose(self) -> ComposeResult:
        with Container(id="help-screen"):
            yield Static(self._markdown)


class InfoModal(_DismissibleModal[None]):
    """Generic Rich-renderable modal shared by /extensions, /tools, /budget.

    Snapshots the renderable at open time — close and reopen to refresh.
    """

    def __init__(self, title: str, body: Any) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Container(id="info-modal"):
            yield Static(Text(self._title, style="bold"), classes="info-title")
            yield Static(self._body, classes="info-body")


class AgentMApp(App[int]):
    CSS_PATH = str(Path(__file__).with_name("textual_app.tcss"))
    # The built-in command palette (Ctrl+P) is disabled — we have our own
    # at Ctrl+R so atom-registered slash commands surface naturally.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "Interrupt", show=True, priority=True),
        Binding("ctrl+d", "force_quit", "Quit", show=True, priority=True),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
        Binding("ctrl+r", "open_palette_binding", "Commands", show=True),
        Binding("ctrl+e", "toggle_tool", "Toggle tool", show=True),
        Binding("pageup", "scroll_page_up", show=False),
        Binding("pagedown", "scroll_page_down", show=False),
    ]

    # Window during which a second Ctrl+C escalates from "cancel" to "quit".
    _CTRL_C_ESCALATE_WINDOW_S = 1.5

    def __init__(
        self,
        config: AgentSessionConfig,
        *,
        theme: str = "dark",
        session: _SessionLike,
        slash_commands: list[SlashCommandEntry],
        extensions: dict[str, ExtensionInstallEvent] | None = None,
        tools: list[ToolEntry] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._session = session
        self._theme_name = theme
        self._slash_commands = slash_commands
        # Snapshots that drive the /extensions and /tools modals. The
        # ``run()`` factory accumulates create-time events into these
        # dicts so the user can ``/extensions`` immediately after
        # launch. Live updates from reload / runtime registration mutate
        # the same dicts via the bus handlers.
        self._extensions: dict[str, ExtensionInstallEvent] = dict(extensions or {})
        self._tools: dict[str, ToolEntry] = {
            entry.name: entry for entry in (tools or [])
        }
        self._budget_state: CostBudgetExceededEvent | None = None
        # Cached on first mount via on_mount; before then it stays None
        # and registry counters live only in the snapshot dicts above.
        self._header: StatusHeader | None = None
        self._prompt_task: asyncio.Task[None] | None = None
        self._flush_timer: Any = None
        # ``turn_index`` from the kernel restarts at 0 on every ``prompt()``
        # call (loop.py: ``for turn_index in range(max_turns)``). Keying
        # purely by ``turn_index`` collides across user messages — the
        # second prompt's turn_index=0 events would be routed back into the
        # first prompt's TurnContainer, so the new assistant text and tool
        # calls render above the new user bubble. The composite key
        # ``(prompt_epoch, turn_index)`` keeps each user-prompt's turns in
        # their own namespace; ``_prompt_epoch`` is bumped from
        # ``submit_input`` and ``handle_api_send_user_message``.
        self._prompt_epoch = 0
        self._root_turns: dict[tuple[int, int], TurnContainer] = {}
        self._child_turns: dict[str, SubagentBlock] = {}
        self._tool_states: dict[str, ToolRenderState] = {}
        self._latest_usage: Usage | None = None
        self._last_assistant_text = ""
        self._history: list[str] = []
        self._history_index: int | None = None
        self._ctrl_c_armed_at: float | None = None
        self._turn_counter = 0
        self._needs_scroll_end = False

    def compose(self) -> ComposeResult:
        # outside-in: status header docks top, input bar docks bottom,
        # conversation log fills the remaining 1fr middle. The Textual
        # ``Footer`` was previously yielded here to surface the binding
        # legend, but it overlapped the prompt input on common terminal
        # sizes and the same legend is reachable via ``/help``.
        yield StatusHeader(id="status-header")
        yield ConversationLog(id="conversation-log")
        with Container(id="input-bar"):
            yield PromptInput()

    def on_mount(self) -> None:
        # Use Textual's built-in theme system instead of our custom
        # .theme-dark / .theme-light classes. The CLI flag still accepts
        # short aliases ("dark", "light") for backwards compat.
        theme_name = _THEME_ALIASES.get(self._theme_name, self._theme_name)
        try:
            self.theme = theme_name
        except Exception:  # noqa: BLE001 — tolerate unknown theme strings
            self.theme = "textual-dark"
        # Keep one custom class on the App so existing tests (and any
        # ad-hoc scripts) can still observe the requested theme name.
        self.add_class(f"theme-{self._theme_name}")
        self.query_one(PromptInput).focus()
        self.refresh_input_height()
        self._header = self.query_one(StatusHeader)
        self._header.model_name = getattr(self._session.model, "id", "?") or "?"
        self._refresh_registry_counters()
        self._flush_timer = self.set_interval(0.05, self.flush_stream_buffers)

    def _refresh_registry_counters(self) -> None:
        if self._header is None:
            return
        loaded = 0
        failed = 0
        for ev in self._extensions.values():
            if ev.phase == "end":
                loaded += 1
            elif ev.phase == "error":
                failed += 1
        self._header.set_registry(loaded=loaded, failed=failed, tools=len(self._tools))

    async def on_unmount(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
        await self._session.shutdown()

    def refresh_input_height(self) -> None:
        input_widget = self.query_one(PromptInput)
        lines = max(1, input_widget.text.count("\n") + 1)
        input_widget.styles.height = min(max(lines + 1, 3), 8)

    async def submit_input(self) -> None:
        input_widget = self.query_one(PromptInput)
        text = input_widget.text
        if not text.strip() or self._prompt_task is not None:
            return
        if await self._handle_builtin_command(text.strip()):
            input_widget.clear()
            self.refresh_input_height()
            return
        self._history.append(text)
        self._history_index = None
        self._ctrl_c_armed_at = None
        # Bump epoch BEFORE awaiting anything — once the prompt task fires
        # the kernel will start emitting events with turn_index=0, and they
        # must land in a fresh ``(epoch, 0)`` slot rather than the previous
        # prompt's slot.
        self._prompt_epoch += 1
        await self._append_user_turn(text)
        input_widget.clear()
        self.refresh_input_height()
        self._prompt_task = asyncio.create_task(self._drive_prompt(text))

    async def _drive_prompt(self, text: str) -> None:
        try:
            await self._session.prompt(text)
        except asyncio.CancelledError:
            self.set_phase("idle")
            self.flush_stream_buffers()
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Prompt failed: {exc}", severity="error")
            self.set_phase("idle")
        finally:
            self._prompt_task = None
            self.query_one(PromptInput).focus()

    async def _append_user_turn(self, text: str) -> None:
        log = self.query_one(ConversationLog)
        await log.mount(UserTurn(text))
        self._needs_scroll_end = True
        log.scroll_end(animate=False)

    def set_phase(self, phase: str) -> None:
        if self._header is not None:
            self._header.phase = phase

    def cycle_history(self, direction: int) -> bool:
        if not self._history:
            return False
        if self._history_index is None:
            self._history_index = len(self._history) - 1 if direction < 0 else 0
        else:
            self._history_index = max(
                0, min(len(self._history) - 1, self._history_index + direction)
            )
        prompt = self.query_one(PromptInput)
        prompt.load_text(self._history[self._history_index])
        prompt.move_cursor(
            (len(prompt.document.lines) - 1, len(prompt.document.lines[-1]))
        )
        self.refresh_input_height()
        return True

    async def _handle_builtin_command(self, text: str) -> bool:
        lowered = text.lower()
        if lowered in _QUIT_COMMANDS:
            self.exit(0)
            return True
        if lowered == "/clear":
            await self.action_clear_log()
            return True
        if lowered == "/help":
            self.push_screen(HelpScreen(self._all_commands()))
            return True
        if lowered == "/copy-last":
            self._copy_last_assistant_text()
            return True
        if lowered == "/extensions":
            self.push_screen(
                InfoModal("Extensions", _build_extensions_table(self._extensions))
            )
            return True
        if lowered == "/tools":
            self.push_screen(InfoModal("Tools", _build_tools_table(self._tools)))
            return True
        if lowered == "/budget":
            self.push_screen(
                InfoModal("Budget", _build_budget_panel(self._budget_state))
            )
            return True
        return False

    def _copy_last_assistant_text(self) -> None:
        if not self._last_assistant_text:
            return
        try:
            import pyperclip  # type: ignore[import-untyped]

            pyperclip.copy(self._last_assistant_text)
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            payload = base64.b64encode(
                self._last_assistant_text.encode("utf-8")
            ).decode("ascii")
            sys.stdout.write(f"\033]52;c;{payload}\a")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            return

    def open_command_palette(self, *, initial: str = "") -> None:
        self.push_screen(
            CommandPaletteScreen(self._all_commands(), initial=initial),
            self._apply_palette_selection,
        )

    def _apply_palette_selection(self, selection: str | None) -> None:
        if selection is None:
            self.query_one(PromptInput).focus()
            return
        prompt = self.query_one(PromptInput)
        prompt.load_text(selection)
        prompt.move_cursor((0, len(selection)))
        self.refresh_input_height()
        prompt.focus()

    def _all_commands(self) -> list[SlashCommandEntry]:
        return sorted(self._slash_commands, key=lambda item: item.name)

    async def action_soft_cancel(self) -> None:
        prompt = self.query_one(PromptInput)
        if self._prompt_task is not None:
            task = self._prompt_task
            self._prompt_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.set_phase("idle")
            prompt.focus()
            return
        if prompt.text:
            prompt.clear()
            self.refresh_input_height()
            return
        # Idle Esc was previously a silent no-op which made users wonder
        # whether the app had stalled. Emit an explicit toast so the key
        # is visibly acknowledged.
        self.notify("Nothing to cancel.", timeout=1.5)
        prompt.focus()

    async def action_interrupt_or_quit(self) -> None:
        now = time.monotonic()
        armed_within_window = (
            self._ctrl_c_armed_at is not None
            and (now - self._ctrl_c_armed_at) < self._CTRL_C_ESCALATE_WINDOW_S
        )
        if self._prompt_task is not None:
            if armed_within_window:
                self.exit(0)
                return
            await self.action_soft_cancel()
            self._ctrl_c_armed_at = now
            self.notify(
                "Cancelled. Press Ctrl+C again within 1.5s to quit.",
                timeout=1.5,
            )
            return
        log = self.query_one(ConversationLog)
        if not log.children:
            self.exit(0)
            return
        if armed_within_window:
            self.exit(0)
            return
        self._ctrl_c_armed_at = now
        self.notify(
            "Press Ctrl+C again within 1.5s to quit.",
            timeout=1.5,
        )

    async def action_force_quit(self) -> None:
        self.exit(0)

    async def action_clear_log(self) -> None:
        log = self.query_one(ConversationLog)
        await log.remove_children()
        self._root_turns.clear()
        self._child_turns.clear()
        self._tool_states.clear()
        self._last_assistant_text = ""
        self._ctrl_c_armed_at = None
        # No need to reset ``_prompt_epoch`` — its only purpose is to keep
        # the dict keys unique across user prompts, and a monotonically
        # growing counter does that whether or not the visual log is
        # cleared.

    def action_open_palette_binding(self) -> None:
        self.open_command_palette(initial="")

    def action_scroll_page_up(self) -> None:
        self.query_one(ConversationLog).scroll_page_up(animate=False)

    def action_scroll_page_down(self) -> None:
        self.query_one(ConversationLog).scroll_page_down(animate=False)

    def action_toggle_tool(self) -> None:
        focused = self.focused
        if isinstance(focused, ToolCallBlock):
            focused.toggle_collapsed()

    def flush_stream_buffers(self) -> None:
        updated = False
        for turn in self._root_turns.values():
            if turn.assistant.text != turn.text_buffer:
                turn.assistant.set_text(turn.text_buffer)
                updated = True
            if turn.thinking.text != turn.thinking_buffer:
                turn.thinking.set_text(turn.thinking_buffer)
                updated = True
            if turn.text_buffer:
                self._last_assistant_text = turn.text_buffer
        try:
            log = self.query_one(ConversationLog)
        except NoMatches:
            return
        if updated or self._needs_scroll_end:
            log.scroll_end(animate=False)
            self._needs_scroll_end = False

    def _ensure_root_turn_sync(self, turn_index: int) -> TurnContainer:
        key = (self._prompt_epoch, turn_index)
        turn = self._root_turns.get(key)
        if turn is not None:
            return turn
        self._turn_counter += 1
        turn = TurnContainer(logical_turn=self._turn_counter)
        self._root_turns[key] = turn
        self.run_worker(self._mount_turn(turn), exclusive=False)
        if self._header is not None:
            self._header.turn_number = self._turn_counter
        return turn

    async def _mount_turn(self, turn: TurnContainer) -> None:
        log = self.query_one(ConversationLog)
        await log.mount(turn)
        self._needs_scroll_end = True
        log.scroll_end(animate=False)

    def _latest_turn(self) -> TurnContainer | None:
        if not self._root_turns:
            return None
        # Tuples sort lexicographically — newer epoch wins, and within the
        # same epoch the higher turn_index wins. Both are what we want.
        return self._root_turns[max(self._root_turns)]

    def _latest_tool_block(self) -> ToolCallBlock | None:
        for key in sorted(self._root_turns, reverse=True):
            turn = self._root_turns[key]
            if turn.tools:
                return next(reversed(turn.tools.values()))
        return None

    def _update_usage(self, usage: Usage | None) -> None:
        if usage is None or self._header is None:
            return
        status = self._header
        status.tokens_in = usage.input_tokens
        status.tokens_out = usage.output_tokens
        model = self._session.model
        pricing = getattr(model, "metadata", {}).get("pricing", (0.0, 0.0))
        status.cost_usd = (usage.input_tokens / 1_000_000.0) * pricing[0] + (
            usage.output_tokens / 1_000_000.0
        ) * pricing[1]

    def _render_child_delta(self, delta: Any) -> None:
        if not self._child_turns:
            return
        child = next(reversed(self._child_turns.values()))
        if isinstance(delta, TextDelta):
            child.add_line(delta.text)
        elif isinstance(delta, ThinkingDelta):
            child.add_line(f"[thinking] {delta.text}")
        elif isinstance(delta, ToolCallStart):
            child.add_line(f"→ {delta.name} …")

    def handle_stream_delta(self, event: StreamDeltaEvent) -> None:
        delta = event.delta
        turn = self._ensure_root_turn_sync(event.turn_index)
        if isinstance(delta, TextDelta):
            turn.text_buffer += delta.text
            self.set_phase("streaming")
            self._needs_scroll_end = True
            if self._child_turns:
                self._render_child_delta(delta)
            return
        if isinstance(delta, ThinkingDelta):
            turn.thinking_buffer += delta.text
            self.set_phase("thinking")
            self._needs_scroll_end = True
            if self._child_turns:
                self._render_child_delta(delta)
            return
        if isinstance(delta, ToolCallStart):
            self._tool_states[delta.id] = ToolRenderState(
                tool_name=delta.name,
                start_ns=time.perf_counter_ns(),
                args_json_fragments=[],
            )
            self.run_worker(turn.ensure_tool(delta.id, delta.name), exclusive=False)
            self._needs_scroll_end = True
            if self._child_turns:
                self._render_child_delta(delta)
            return
        if isinstance(delta, ToolCallArgsDelta):
            state = self._tool_states.get(delta.id)
            if state is not None and state.args_json_fragments is not None:
                state.args_json_fragments.append(delta.args_json_delta)
            return
        if isinstance(delta, MessageEnd):
            self._latest_usage = delta.message.usage
            self._last_assistant_text = _assistant_text(delta.message)

    def handle_tool_call(self, event: ToolCallEvent) -> None:
        turn = self._latest_turn()
        if turn is None:
            return
        state = self._tool_states.get(event.tool_call_id)
        if state is None:
            state = ToolRenderState(
                tool_name=event.tool_name, start_ns=time.perf_counter_ns()
            )
            self._tool_states[event.tool_call_id] = state
        state.args = dict(event.args)
        self.set_phase("tool")

        async def _apply() -> None:
            block = await turn.ensure_tool(event.tool_call_id, event.tool_name)
            block.set_pending_args(dict(event.args), "")

        self.run_worker(_apply(), exclusive=False)
        self._needs_scroll_end = True

    def handle_tool_result(self, event: ToolResultEvent) -> None:
        turn = self._latest_turn()
        if turn is None:
            return
        state = self._tool_states.get(event.tool_call_id)
        duration_ms = 0
        if state is not None:
            duration_ms = max(
                0, int((time.perf_counter_ns() - state.start_ns) / 1_000_000)
            )

        async def _apply() -> None:
            block = await turn.ensure_tool(event.tool_call_id, event.tool_name)
            if state is not None and state.args is not None:
                block.args = dict(state.args)
            block.set_result(result=event.result, duration_ms=duration_ms)

        self.run_worker(_apply(), exclusive=False)
        self.set_phase("tool")
        self._needs_scroll_end = True

    def handle_llm_start(self, event: LlmRequestStartEvent) -> None:
        self._ensure_root_turn_sync(event.turn_index)
        self.set_phase("thinking")
        if self._header is not None:
            self._header.model_name = (
                event.model_id or getattr(self._session.model, "id", "?") or "?"
            )

    def handle_llm_end(self, event: LlmRequestEndEvent) -> None:
        del event
        self.flush_stream_buffers()
        turn = self._latest_turn()
        if turn is not None and turn.thinking.text.strip():
            turn.thinking.collapse()
        self._update_usage(self._latest_usage)
        self.set_phase("idle")

    def handle_child_start(self, event: ChildSessionStartEvent) -> None:
        turn = self._latest_turn()
        if turn is None:
            return
        block = SubagentBlock(event.child_session_id, event.purpose)
        turn.subagents[event.child_session_id] = block
        self._child_turns[event.child_session_id] = block

        async def _apply() -> None:
            await turn.mount(block)

        self.run_worker(_apply(), exclusive=False)
        self.set_phase("subagent")
        self._needs_scroll_end = True

    def handle_child_end(self, event: ChildSessionEndEvent) -> None:
        block = self._child_turns.pop(event.child_session_id, None)
        if block is not None:
            block.mark_error(event.error)
        self.set_phase("idle")
        self._needs_scroll_end = True

    def handle_extension_install(self, event: ExtensionInstallEvent) -> None:
        self._extensions[event.module_path] = event
        # ``start`` does not change end/error counters — skip the
        # refresh + DOM touch for it.
        if event.phase != "start":
            self._refresh_registry_counters()
        if event.phase == "error":
            self.notify(
                f"Extension install failed: {event.module_path}: {event.error}",
                severity="error",
                timeout=5,
            )

    def handle_api_register(self, event: ApiRegisterEvent) -> None:
        if event.kind == "tool":
            tool = event.payload
            self._tools[event.name] = ToolEntry(
                name=event.name,
                description=str(getattr(tool, "description", "")),
                source=event.extension,
            )
            self._refresh_registry_counters()
        elif event.kind == "command" and isinstance(event.payload, CommandSpec):
            entry = SlashCommandEntry(
                name=f"/{event.name}",
                description=event.payload.description,
                source=event.extension,
            )
            self._slash_commands = [
                e for e in self._slash_commands if e.name != entry.name
            ] + [entry]

    def handle_extension_reload(self, event: ExtensionReloadEvent) -> None:
        """Self-modification (``trigger=agent`` or ``propose_change_approved``)
        gets a stronger toast severity than human reloads — the framework's
        signature behavior should never be silent."""

        if event.error:
            self.notify(
                f"Atom reload failed: {event.name} ({event.trigger}): {event.error}",
                severity="error",
                timeout=6,
            )
            return
        is_self_modify = event.trigger in _SELF_MODIFY_TRIGGERS
        prefix = "★ self-modify" if is_self_modify else "atom reload"
        old = (event.old_hash or "—")[:8]
        new = event.new_hash[:8]
        severity: Literal["warning", "information"] = (
            "warning" if is_self_modify else "information"
        )
        self.notify(
            f"{prefix}: {event.name} ({event.trigger}) {old} → {new}",
            severity=severity,
            timeout=4,
        )

    def handle_cost_budget_exceeded(self, event: CostBudgetExceededEvent) -> None:
        """Latch budget-exceeded state. The next ``prompt`` will short-
        circuit with ``stop_reason='budget'`` (see ``cost_budget`` atom),
        and the user needs to know *now*."""

        self._budget_state = event
        if self._header is not None:
            self._header.set_budget_exceeded(True)
        self.notify(
            f"Budget exceeded: ${event.used:.4f} / ${event.limit:.4f} {event.currency}. "
            "Next prompt will halt with stop_reason='budget'.",
            severity="error",
            timeout=8,
        )

    def handle_api_send_user_message(self, event: ApiSendUserMessageEvent) -> None:
        """The kernel treats an extension-injected message exactly like
        a user-typed one; rendering it with a distinct gutter is the
        only signal the user has that they did not, in fact, type it."""

        text = event.content if isinstance(event.content, str) else repr(event.content)

        async def _mount() -> None:
            log = self.query_one(ConversationLog)
            await log.mount(UserTurn(text, injected_from=event.extension))
            log.scroll_end(animate=False)

        self.run_worker(_mount(), exclusive=False)
        self._needs_scroll_end = True


async def run(config: AgentSessionConfig, *, theme: str = "dark") -> int:
    bus = config.bus
    if bus is None:
        from agentm.core.abi import EventBus

        bus = EventBus()

    # ``AgentSession.create`` triggers extension load which fires
    # ``api_register`` (kind=command/tool) and ``extension_install`` on
    # the bus. We must subscribe BEFORE create to capture those — the
    # AgentMApp instance does not exist yet. The accumulators are then
    # handed to ``__init__`` so /extensions and /tools modals are
    # useful from the first keystroke. Each ``bus.on`` returns an
    # unsubscribe callable; we hold them so the live ``app.handle_*``
    # subscribers below are the only ones that keep firing.

    slash_commands: dict[str, SlashCommandEntry] = {
        name: SlashCommandEntry(name=name, description=description, source="builtin")
        for name, description in _BUILTIN_COMMANDS.items()
    }
    tools: dict[str, ToolEntry] = {}
    extensions: dict[str, ExtensionInstallEvent] = {}

    def _capture_register(event: ApiRegisterEvent) -> None:
        if event.kind == "command" and isinstance(event.payload, CommandSpec):
            slash_commands[f"/{event.name}"] = SlashCommandEntry(
                name=f"/{event.name}",
                description=event.payload.description,
                source=event.extension,
            )
        elif event.kind == "tool":
            tools[event.name] = ToolEntry(
                name=event.name,
                description=str(getattr(event.payload, "description", "")),
                source=event.extension,
            )

    def _capture_extension_install(event: ExtensionInstallEvent) -> None:
        extensions[event.module_path] = event

    unsub_register = bus.on(ApiRegisterEvent.CHANNEL, _capture_register)
    unsub_install = bus.on(ExtensionInstallEvent.CHANNEL, _capture_extension_install)

    session_cfg = config.with_bus(bus)
    try:
        session = await AgentSession.create(session_cfg)
    finally:
        # Drop the create-time accumulators no matter what — leaving
        # them subscribed would mutate ``slash_commands`` / ``tools`` /
        # ``extensions`` for the rest of the session even though those
        # dicts are never read again.
        unsub_register()
        unsub_install()

    app = AgentMApp(
        session_cfg,
        theme=theme,
        session=session,
        slash_commands=list(slash_commands.values()),
        extensions=extensions,
        tools=list(tools.values()),
    )

    # Live subscriptions. Hot reloads, late tool registration, budget
    # overflow, and extension-injected user messages all happen after
    # create — these handlers carry the live state through.

    bus.on(StreamDeltaEvent.CHANNEL, app.handle_stream_delta)
    bus.on(ToolCallEvent.CHANNEL, app.handle_tool_call)
    bus.on(ToolResultEvent.CHANNEL, app.handle_tool_result)
    bus.on(LlmRequestStartEvent.CHANNEL, app.handle_llm_start)
    bus.on(LlmRequestEndEvent.CHANNEL, app.handle_llm_end)
    bus.on(ChildSessionStartEvent.CHANNEL, app.handle_child_start)
    bus.on(ChildSessionEndEvent.CHANNEL, app.handle_child_end)
    bus.on(ExtensionInstallEvent.CHANNEL, app.handle_extension_install)
    bus.on(ApiRegisterEvent.CHANNEL, app.handle_api_register)
    bus.on(ExtensionReloadEvent.CHANNEL, app.handle_extension_reload)
    bus.on(CostBudgetExceededEvent.CHANNEL, app.handle_cost_budget_exceeded)
    bus.on(ApiSendUserMessageEvent.CHANNEL, app.handle_api_send_user_message)

    result = await app.run_async()
    return 0 if result is None else int(result)


def _assistant_text(message: AssistantMessage) -> str:
    chunks: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            chunks.append(block.text)
    return "\n".join(chunks)


def _tool_result_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        return ""
    return "\n\n".join(parts)


def _looks_like_markdown(text: str) -> bool:
    return "```" in text or "# " in text or "- " in text


def _looks_like_diff(tool_name: str, text: str) -> bool:
    if tool_name == "tool_edit":
        return True
    stripped = text.lstrip()
    return stripped.startswith("--- ") or stripped.startswith("+++ ")


def _render_tool_result(tool_name: str, text: str) -> Any:
    if _looks_like_diff(tool_name, text):
        return Syntax(text, "diff", word_wrap=True)
    if _looks_like_markdown(text):
        return Markdown(text)
    return Text(text)


def _format_args_preview(args: dict[str, Any]) -> str:
    return _dump_json(args)


def _format_full_args(args: dict[str, Any]) -> str:
    return _dump_json(args, indent=2)


def _format_args_inline(args: dict[str, Any]) -> str:
    """Render args as a one-line preview to embed in a Collapsible title.

    Single-arg → just the value (the keyword is implied by the tool name);
    multi-arg → ``key=val, key=val`` with each value truncated. Empty args
    render as ``…`` so the title always reads ``tool_name(…)`` rather than
    ``tool_name()`` (which looks like a no-arg call).
    """

    if not args:
        return "…"
    if len(args) == 1:
        return _truncate(_arg_value_str(next(iter(args.values()))), 64)
    parts = [f"{k}={_truncate(_arg_value_str(v), 16)}" for k, v in args.items()]
    return _truncate(", ".join(parts), 80)


def _arg_value_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _dump_json(value)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _line_count(text: str) -> int:
    return 0 if not text else text.count("\n") + 1


def _dump_json(value: Any, *, indent: int | None = None) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )
    except TypeError:
        return repr(value)


# ---------------------------------------------------------------------------
# Modal renderable builders (control + observability surface).
#
# These produce Rich renderables for the generic InfoModal so /extensions,
# /tools, and /budget can render uniform tables without each carrying its
# own Textual subclass.
# ---------------------------------------------------------------------------


_INSTALL_PHASE_GLYPH: dict[str, tuple[str, str]] = {
    "start": ("⏳", "yellow"),
    "end": ("✓", "green"),
    "error": ("✗", "red"),
}


def _empty_state(message: str) -> Text:
    return Text(message, style="dim italic")


def _build_extensions_table(extensions: dict[str, ExtensionInstallEvent]) -> Any:
    if not extensions:
        return _empty_state(
            "No extensions tracked yet. Either none loaded, or the bus "
            "subscribers were wired after AgentSession.create."
        )
    table = RichTable(show_header=True, header_style="bold", expand=True)
    table.add_column("Status", width=8)
    table.add_column("Module", overflow="fold")
    table.add_column("Error", overflow="fold")
    for module_path in sorted(extensions):
        event = extensions[module_path]
        glyph, style = _INSTALL_PHASE_GLYPH.get(event.phase, ("?", "white"))
        table.add_row(
            Text(f"{glyph} {event.phase}", style=style),
            module_path,
            event.error or "",
        )
    return table


def _build_tools_table(tools: dict[str, ToolEntry]) -> Any:
    if not tools:
        return _empty_state(
            "No tools registered. The agent has no callable surface — "
            "check that the scenario or auto-discovered atoms loaded."
        )
    table = RichTable(show_header=True, header_style="bold", expand=True)
    table.add_column("Tool", width=24)
    table.add_column("Source", width=20)
    table.add_column("Description", overflow="fold")
    for name in sorted(tools):
        entry = tools[name]
        table.add_row(
            Text(entry.name, style="bold yellow"),
            entry.source,
            entry.description or "—",
        )
    return table


def _build_budget_panel(state: CostBudgetExceededEvent | None) -> Any:
    if state is None:
        return _empty_state(
            "Budget OK. No cost_budget_exceeded event has fired in this "
            "session — either the cost_budget atom is not loaded, or "
            "spend has not crossed the configured limit yet."
        )
    table = RichTable(show_header=False, expand=False)
    table.add_column(width=12)
    table.add_column()
    table.add_row(Text("status", style="bold"), Text("EXCEEDED", style="bold red"))
    table.add_row("used", f"{state.used:.6f} {state.currency}")
    table.add_row("limit", f"{state.limit:.6f} {state.currency}")
    table.add_row(
        "next prompt",
        Text(
            "will halt with stop_reason='budget'",
            style="red italic",
        ),
    )
    return table


__all__ = ["AgentMApp", "run"]
