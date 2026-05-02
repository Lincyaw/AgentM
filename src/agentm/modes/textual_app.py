from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
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
from agentm.extensions.builtin.cost_budget import _PRICING as _COST_PRICING
from agentm.harness import AgentSession, AgentSessionConfig
from agentm.harness.events import (
    ApiRegisterEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    ExtensionInstallEvent,
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
class ToolRenderState:
    tool_name: str
    start_ns: int
    args: dict[str, Any] | None = None
    args_json_fragments: list[str] | None = None


class ConversationLog(VerticalScroll):
    pass


class StatusLine(Static):
    model_name = reactive("?")
    current_text = reactive("? · turn 0 · in: 0 · out: 0 · $0.000 · idle")
    turn_number = reactive(0)
    tokens_in = reactive(0)
    tokens_out = reactive(0)
    cost_usd = reactive(0.0)
    phase = reactive("idle")

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

    def _refresh(self) -> None:
        self.current_text = (
            f"{self.model_name} · turn {self.turn_number} · in: {self.tokens_in}"
            f" · out: {self.tokens_out} · ${self.cost_usd:.3f} · {self.phase}"
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


class ThinkingBlock(Collapsible):
    def __init__(self) -> None:
        self.body = Static(classes="thinking-body")
        super().__init__(self.body, title="thinking", collapsed=False, classes="thinking")
        self.visible = False
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text
        stripped = text.strip()
        self.visible = bool(stripped)
        self.body.update(Text(stripped, style="dim italic"))


class ToolCallBlock(Collapsible):
    can_focus = True

    def __init__(self, tool_call_id: str, tool_name: str) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.body = Static(classes="tool-body")
        super().__init__(self.body, title=f"→ {tool_name} …  [pending]", collapsed=False)
        self.add_class("tool-call")
        self.result_text = ""
        self.args: dict[str, Any] = {}
        self.body_kind = "text"
        self.ok: bool | None = None
        self.duration_ms = 0

    def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed

    def set_pending_args(self, args: dict[str, Any], args_preview: str) -> None:
        self.args = dict(args)
        self.title = f"→ {self.tool_name} {args_preview}  [pending]"
        self._render_body(None)

    def set_result(self, *, result: Any, duration_ms: int) -> None:
        self.duration_ms = duration_ms
        self.ok = not getattr(result, "is_error", False)
        self.result_text = _tool_result_text(result)
        status = "✓" if self.ok else "✗"
        args_preview = _truncate(_format_args_preview(self.args), 48) if self.args else "{}"
        self.title = f"→ {self.tool_name} {args_preview}  [{status} {duration_ms}ms]"
        self.collapsed = self.ok and _line_count(self.result_text) < 20
        self._render_body(result)
        if self.ok:
            self.remove_class("failed")
        else:
            self.add_class("failed")

    def _render_body(self, result: Any | None) -> None:
        args_text = _format_full_args(self.args)
        result_text = self.result_text if result is not None else ""
        if result is None:
            renderable: Any = Text(f"args\n{args_text}", style="yellow")
            self.body_kind = "text"
        else:
            result_renderable = _render_tool_result(self.tool_name, result_text)
            self.body_kind = type(result_renderable).__name__
            renderable = Group(
                Text("args", style="yellow"),
                Text(args_text),
                Text("result", style="green" if not getattr(result, "is_error", False) else "red"),
                result_renderable,
            )
        self.body.update(renderable)


class SubagentBlock(Static):
    def __init__(self, child_session_id: str, purpose: str) -> None:
        super().__init__(classes="subagent")
        self.child_session_id = child_session_id
        self.purpose = purpose
        self.lines: list[str] = [f"subagent: {purpose}"]
        self.failed = False
        self._refresh()

    def add_line(self, text: str) -> None:
        if text:
            self.lines.append(text)
            self._refresh()

    def mark_error(self, error: str | None) -> None:
        self.failed = bool(error)
        if error:
            self.lines.append(f"error: {error}")
        self.set_class(self.failed, "failed")
        self._refresh()

    def _refresh(self) -> None:
        border = "red" if self.failed else "cyan"
        self.update(
            Panel(
                Markdown("\n\n".join(self.lines)),
                title=f"↳ {self.purpose}",
                border_style=border,
            )
        )


class UserTurn(Static):
    def __init__(self, text: str) -> None:
        super().__init__(classes="user-turn")
        self.update(Panel(Markdown(text), title="user", border_style="blue"))


class TurnContainer(Vertical):
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
        self.tab_behavior = "indent"

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
        if event.character == "/" and not self.text and self.cursor_location == (0, 0):
            event.prevent_default()
            event.stop()
            app.open_command_palette(initial="")
            return
        app.refresh_input_height()


class CommandPaletteScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, commands: list[SlashCommandEntry], *, initial: str = "") -> None:
        super().__init__()
        self._commands = commands
        self._initial = initial
        self._filtered = list(commands)

    def compose(self) -> ComposeResult:
        with Container(id="command-palette"):
            yield Input(value=self._initial, placeholder="Filter commands", id="command-filter")
            yield OptionList(id="command-options")

    def on_mount(self) -> None:
        self._refresh_options(self._initial)
        self.query_one("#command-filter", Input).focus()

    def action_cancel(self) -> None:
        if self.app.screen is not self:
            return
        try:
            self.dismiss(None)
        except ScreenStackError:
            return

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
                if needle in command.name.lower() or needle in command.description.lower()
            ]
        else:
            self._filtered = list(self._commands)
        options = self.query_one("#command-options", OptionList)
        options.clear_options()
        for command in self._filtered:
            options.add_option(Option(f"{command.name} — {command.description}", id=command.name))
        if options.option_count:
            options.highlighted = 0


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "cancel", "Close")]

    def __init__(self, commands: list[SlashCommandEntry]) -> None:
        super().__init__()
        command_lines = "\n".join(f"- `{entry.name}` — {entry.description}" for entry in commands)
        self._markdown = Markdown(
            "# Help\n\n"
            "## Keys\n"
            "- `Enter` submit\n"
            "- `Shift+Enter` newline\n"
            "- `Esc` soft-cancel\n"
            "- `Ctrl+C` interrupt / quit\n"
            "- `Ctrl+D` quit\n"
            "- `Ctrl+L` clear log\n"
            "- `Ctrl+R` command palette\n"
            "- `Tab` focus toggle\n"
            "- `PageUp` / `PageDown` scroll\n"
            "- `Up` / `Down` history when input is empty\n"
            "- `Ctrl+E` toggle focused tool block\n\n"
            "## Slash commands\n"
            f"{command_lines}"
        )

    def compose(self) -> ComposeResult:
        with Container(id="help-screen"):
            yield Static(self._markdown)

    def action_cancel(self) -> None:
        if self.app.screen is not self:
            return
        try:
            self.dismiss(None)
        except ScreenStackError:
            return


class AgentMApp(App[int]):
    CSS_PATH = str(Path(__file__).with_name("textual_app.tcss"))
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", show=False),
        Binding("ctrl+d", "force_quit", show=False),
        Binding("ctrl+l", "clear_log", show=False),
        Binding("ctrl+r", "open_palette_binding", show=False),
        Binding("tab", "toggle_focus", show=False),
        Binding("pageup", "scroll_page_up", show=False),
        Binding("pagedown", "scroll_page_down", show=False),
        Binding("ctrl+e", "toggle_tool", show=False),
    ]

    def __init__(
        self,
        config: AgentSessionConfig,
        *,
        theme: str = "dark",
        session: _SessionLike,
        slash_commands: list[SlashCommandEntry],
    ) -> None:
        super().__init__()
        self.config = config
        self._session = session
        self._theme_name = theme
        self._slash_commands = slash_commands
        self._prompt_task: asyncio.Task[None] | None = None
        self._flush_timer: Any = None
        self._root_turns: dict[int, TurnContainer] = {}
        self._child_turns: dict[str, SubagentBlock] = {}
        self._tool_states: dict[str, ToolRenderState] = {}
        self._latest_usage: Usage | None = None
        self._last_assistant_text = ""
        self._history: list[str] = []
        self._history_index: int | None = None
        self._ctrl_c_armed = False
        self._turn_counter = 0
        self._needs_scroll_end = False

    def compose(self) -> ComposeResult:
        with Vertical(id="app-root"):
            yield ConversationLog(id="conversation-log")
            with Container(id="input-bar"):
                yield PromptInput()
            yield StatusLine(id="status-line")

    def on_mount(self) -> None:
        self.query_one(PromptInput).focus()
        self.refresh_input_height()
        self.query_one(StatusLine).model_name = getattr(self._session.model, "id", "?") or "?"
        if self._theme_name == "light":
            self.add_class("theme-light")
        else:
            self.add_class("theme-dark")
        self._flush_timer = self.set_interval(0.05, self.flush_stream_buffers)

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
        self._ctrl_c_armed = False
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
        self.query_one(StatusLine).phase = phase

    def cycle_history(self, direction: int) -> bool:
        if not self._history:
            return False
        if self._history_index is None:
            self._history_index = len(self._history) - 1 if direction < 0 else 0
        else:
            self._history_index = max(0, min(len(self._history) - 1, self._history_index + direction))
        prompt = self.query_one(PromptInput)
        prompt.load_text(self._history[self._history_index])
        prompt.move_cursor((len(prompt.document.lines) - 1, len(prompt.document.lines[-1])))
        self.refresh_input_height()
        return True

    async def _handle_builtin_command(self, text: str) -> bool:
        lowered = text.lower()
        if lowered in {"/quit", "/exit", "/q"}:
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
            payload = base64.b64encode(self._last_assistant_text.encode("utf-8")).decode("ascii")
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
        prompt.focus()

    async def action_interrupt_or_quit(self) -> None:
        if self._prompt_task is not None:
            await self.action_soft_cancel()
            return
        log = self.query_one(ConversationLog)
        if not log.children:
            self.exit(0)
            return
        if self._ctrl_c_armed:
            self.exit(0)
            return
        self._ctrl_c_armed = True
        self.notify("Press Ctrl+C again to quit.")

    async def action_force_quit(self) -> None:
        self.exit(0)

    async def action_clear_log(self) -> None:
        log = self.query_one(ConversationLog)
        await log.remove_children()
        self._root_turns.clear()
        self._child_turns.clear()
        self._tool_states.clear()
        self._last_assistant_text = ""
        self._ctrl_c_armed = False

    def action_open_palette_binding(self) -> None:
        self.open_command_palette(initial="")

    def action_toggle_focus(self) -> None:
        focused = self.focused
        if isinstance(focused, PromptInput):
            target = self._latest_tool_block()
            if target is not None:
                target.focus()
            else:
                self.query_one(ConversationLog).focus()
            return
        self.query_one(PromptInput).focus()

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
        turn = self._root_turns.get(turn_index)
        if turn is not None:
            return turn
        self._turn_counter += 1
        turn = TurnContainer(logical_turn=self._turn_counter)
        self._root_turns[turn_index] = turn
        self.run_worker(self._mount_turn(turn), exclusive=False)
        self.query_one(StatusLine).turn_number = self._turn_counter
        return turn

    async def _mount_turn(self, turn: TurnContainer) -> None:
        log = self.query_one(ConversationLog)
        await log.mount(turn)
        self._needs_scroll_end = True
        log.scroll_end(animate=False)

    def _latest_turn(self) -> TurnContainer | None:
        if not self._root_turns:
            return None
        return self._root_turns[max(self._root_turns)]

    def _latest_tool_block(self) -> ToolCallBlock | None:
        for turn_index in sorted(self._root_turns, reverse=True):
            turn = self._root_turns[turn_index]
            if turn.tools:
                return next(reversed(turn.tools.values()))
        return None

    def _update_usage(self, usage: Usage | None) -> None:
        if usage is None:
            return
        status = self.query_one(StatusLine)
        status.tokens_in = usage.input_tokens
        status.tokens_out = usage.output_tokens
        model = self._session.model
        pricing = _COST_PRICING.get(getattr(model, "provider", ""), (0.0, 0.0))
        status.cost_usd = (
            (usage.input_tokens / 1_000_000.0) * pricing[0]
            + (usage.output_tokens / 1_000_000.0) * pricing[1]
        )

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
            turn.thinking.visible = True
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
            state = ToolRenderState(tool_name=event.tool_name, start_ns=time.perf_counter_ns())
            self._tool_states[event.tool_call_id] = state
        state.args = dict(event.args)
        self.set_phase("tool")

        async def _apply() -> None:
            block = await turn.ensure_tool(event.tool_call_id, event.tool_name)
            block.set_pending_args(dict(event.args), _truncate(_format_args_preview(event.args), 48))

        self.run_worker(_apply(), exclusive=False)
        self._needs_scroll_end = True

    def handle_tool_result(self, event: ToolResultEvent) -> None:
        turn = self._latest_turn()
        if turn is None:
            return
        state = self._tool_states.get(event.tool_call_id)
        duration_ms = 0
        if state is not None:
            duration_ms = max(0, int((time.perf_counter_ns() - state.start_ns) / 1_000_000))

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
        model_id = event.model_id or getattr(self._session.model, "id", "?") or "?"
        self.query_one(StatusLine).model_name = model_id

    def handle_llm_end(self, event: LlmRequestEndEvent) -> None:
        del event
        self.flush_stream_buffers()
        turn = self._latest_turn()
        if turn is not None and turn.thinking.text.strip():
            turn.thinking.collapsed = True
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
        if event.phase == "error":
            self.notify(
                f"Extension install failed: {event.module_path}: {event.error}",
                severity="error",
                timeout=5,
            )


async def run(config: AgentSessionConfig, *, theme: str = "dark") -> int:
    bus = config.bus
    if bus is None:
        from agentm.core.abi import EventBus

        bus = EventBus()

    slash_commands: dict[str, SlashCommandEntry] = {
        name: SlashCommandEntry(name=name, description=description, source="builtin")
        for name, description in _BUILTIN_COMMANDS.items()
    }

    def _capture_command(event: ApiRegisterEvent) -> None:
        if event.kind != "command" or not isinstance(event.payload, CommandSpec):
            return
        slash_commands[f"/{event.name}"] = SlashCommandEntry(
            name=f"/{event.name}",
            description=event.payload.description,
            source=event.extension,
        )

    bus.on("api_register", _capture_command)
    session_cfg = AgentSessionConfig(**{**config.__dict__, "bus": bus})
    session = await AgentSession.create(session_cfg)
    app = AgentMApp(
        session_cfg,
        theme=theme,
        session=session,
        slash_commands=list(slash_commands.values()),
    )

    bus.on("stream_delta", app.handle_stream_delta)
    bus.on("tool_call", app.handle_tool_call)
    bus.on("tool_result", app.handle_tool_result)
    bus.on("llm_request_start", app.handle_llm_start)
    bus.on("llm_request_end", app.handle_llm_end)
    bus.on("child_session_start", app.handle_child_start)
    bus.on("child_session_end", app.handle_child_end)
    bus.on("extension_install", app.handle_extension_install)

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


__all__ = ["AgentMApp", "run"]
