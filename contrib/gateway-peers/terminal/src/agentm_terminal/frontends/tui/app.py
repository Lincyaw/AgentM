"""``AgentMTui`` — the Textual app. Composition, bindings, lifecycle.

The app drains the client's outbound stream into the :class:`Router`, runs a
20 Hz flush of the active turn's stream buffer, and owns input + bindings. It
holds the wire client so the router's callbacks (mount, status, toast, send)
all funnel through here. See ``.claude/designs/textual-tui.md`` §5.
"""

from __future__ import annotations

import asyncio
import time

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.dom import DOMNode
from textual.widgets import Collapsible, Footer, OptionList, TextArea

from ...client import TerminalClient
from .router import Router
from .state import Catalog, StatusModel
from .theme import CSS_PATH, Phase, resolve_theme
from .widgets import CommandSuggestions, PromptInput, StatusBar, Toast, UserTurn

# The only slash command the TUI handles itself: /clear wipes the rendered
# transcript (inherently client-side) AND cold-resets the gateway session.
# Everything else is a pluggable command registered by an atom/gateway builtin
# and surfaced via the session's command list — the TUI never hardcodes those.
# Pure client conveniences (clear screen, copy, quit) are keybindings, not
# slash commands.
_LOCAL_COMMANDS = ("/clear",)


class AgentMTui(App[int]):
    CSS_PATH = str(CSS_PATH)
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "interrupt", priority=True, show=True),
        Binding("ctrl+d", "quit_app", "quit", priority=True, show=True),
        Binding("ctrl+l", "clear_log", "clear", show=True),
        Binding("ctrl+y", "copy_last", "copy", show=True),
        Binding("ctrl+e", "toggle_tool", "expand", show=True),
        Binding("escape", "cancel", "cancel", show=False),
    ]

    def __init__(
        self,
        *,
        client: TerminalClient,
        sender_id: str,
        chat_id: str,
        theme: str = "dark",
    ) -> None:
        super().__init__()
        self._client = client
        self._sender_id = sender_id
        self._chat_id = chat_id
        self._theme_name = resolve_theme(theme)
        self.status = StatusModel()
        self.catalog = Catalog()
        self._router = Router(self)
        self._consumer: asyncio.Task[None] | None = None
        self._history: list[str] = []
        self._hist_idx = 0  # cursor into _history (== len means "fresh line")
        self._last_text = ""  # last assistant text, for /copy-last
        self._ctrlc_ts = 0.0
        # True from prompt-submit until the loop's agent_end; gates Esc-interrupt.
        self._in_flight = False

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        yield VerticalScroll(id="transcript")
        # The input and its autocomplete share one bottom-docked column so the
        # suggestions render *below* the input (normal top-to-bottom flow),
        # inline — not a floating modal. Footer still pins to the very bottom.
        with Vertical(id="input-dock"):
            yield PromptInput(id="prompt-input", show_line_numbers=False)
            yield CommandSuggestions(id="suggestions")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self.theme = self._theme_name
        except Exception:  # noqa: BLE001 — invalid theme name; keep the default
            pass
        self.show_status()
        self.query_one("#suggestions", CommandSuggestions).display = False
        self.query_one("#prompt-input", PromptInput).focus()
        self._consumer = asyncio.create_task(self._consume(), name="tui-consume")
        self.set_interval(0.05, self._flush_active)

    async def _consume(self) -> None:
        try:
            async for body in self._client.outbound():
                try:
                    await self._router.dispatch(body)
                except Exception as exc:  # noqa: BLE001 — a bad frame must not crash the UI
                    self.log(f"dispatch failed: body={body!r} exc={exc!r}")
        finally:
            self.exit(0)

    async def _flush_active(self) -> None:
        turn = self._router.active
        if turn is not None and await turn.flush():
            # Only scroll when flush actually applied new content — an
            # unconditional scroll_end every tick forces a repaint at the
            # timer rate even during an idle/slow-thinking stretch.
            self.scroll_end()

    # --- router callbacks (the app is the router's view handle) --------

    @property
    def transcript(self) -> VerticalScroll:
        return self.query_one("#transcript", VerticalScroll)

    async def mount_widget(self, widget: object) -> None:
        await self.transcript.mount(widget)  # type: ignore[arg-type]

    def set_phase(self, phase: Phase) -> None:
        self.status.phase = phase
        self.show_status()

    def mark_idle(self) -> None:
        """The loop ended (agent_end). Clear the in-flight gate + go idle."""
        self._in_flight = False
        self.set_phase("idle")

    def show_status(self) -> None:
        self.query_one("#status-bar", StatusBar).show(self.status)

    def toast(self, text: str, *, variant: str = "info") -> None:
        self.mount(Toast(text, variant=variant))

    def scroll_end(self) -> None:
        self.transcript.scroll_end(animate=False)

    def note_assistant_text(self, text: str) -> None:
        """Record the latest assistant text so /copy-last can yank it."""
        if text.strip():
            self._last_text = text

    async def send_button(self, value: str) -> None:
        await self._send(
            {
                "channel": "terminal",
                "sender_id": self._sender_id,
                "chat_id": self._chat_id,
                "content": f"[button: {value}]",
                "button_value": value,
            }
        )

    async def send_interrupt(self) -> None:
        """Out-of-band interrupt: preempt the in-flight prompt. ``control``
        keeps it off the conversational path (gateway routes it to
        AgentSession.interrupt(), not a new turn)."""
        await self._send(
            {
                "channel": "terminal",
                "sender_id": self._sender_id,
                "chat_id": self._chat_id,
                "content": "",
                "control": "interrupt",
            }
        )

    # --- input ----------------------------------------------------------

    @on(PromptInput.Submitted)
    async def _on_submit(self, msg: PromptInput.Submitted) -> None:
        inp = self.query_one("#prompt-input", PromptInput)
        text = msg.text.strip()
        inp.text = ""
        if not text:
            return
        self._history.append(text)
        self._hist_idx = len(self._history)  # reset history cursor to the fresh line
        # Slash commands (not ``//path``): TUI-local ones are handled here;
        # everything else is forwarded to the gateway's command router.
        if text.startswith("/") and not text.startswith("//"):
            if await self._run_slash(text):
                return
        await self.mount_widget(UserTurn(text))
        self.scroll_end()
        self._in_flight = True
        await self._send(
            {
                "channel": "terminal",
                "sender_id": self._sender_id,
                "chat_id": self._chat_id,
                "content": text,
            }
        )

    async def _run_slash(self, text: str) -> bool:
        """Handle a TUI-local slash command. Returns True if handled here,
        False if it should be forwarded to the gateway. Only /clear is local
        (it must wipe the client-rendered transcript); all other slash commands
        are pluggable and forwarded to the gateway/atom command router."""
        name = text.split(maxsplit=1)[0]
        if name == "/clear":
            await self._clear_and_reset()
            return True
        return False

    def action_copy_last(self) -> None:
        if not self._last_text:
            self.toast("nothing to copy yet")
            return
        # OSC 52 clipboard write — works over SSH/tmux without a clipboard dep.
        self.copy_to_clipboard(self._last_text)
        self.toast("copied last reply")

    async def _send(self, body: dict[str, object]) -> None:
        try:
            await self._client.send_inbound(body)
        except Exception:  # noqa: BLE001
            self.toast("failed to send to gateway", variant="warn")

    # --- command autocomplete ------------------------------------------

    @property
    def suggestions(self) -> CommandSuggestions:
        return self.query_one("#suggestions", CommandSuggestions)

    def _command_candidates(self) -> list[str]:
        # Dynamic source: /clear (the one client-local command) + everything an
        # atom/gateway registered, surfaced via the catalog. Nothing hardcoded.
        return sorted({*_LOCAL_COMMANDS, *self.catalog.commands})

    @on(TextArea.Changed, "#prompt-input")
    def _refresh_suggestions(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        typing_command = (
            text.startswith("/")
            and not text.startswith("//")
            and " " not in text
            and "\n" not in text
        )
        matches = (
            [c for c in self._command_candidates() if c.startswith(text)]
            if typing_command
            else []
        )
        self.suggestions.populate(matches)

    @on(PromptInput.Complete)
    def _on_complete(self, _msg: PromptInput.Complete) -> None:
        self._apply_completion()

    @on(OptionList.OptionSelected, "#suggestions")
    def _on_suggestion_clicked(self, event: OptionList.OptionSelected) -> None:
        self._apply_completion(str(event.option.prompt))

    def _apply_completion(self, command: str | None = None) -> None:
        sug = self.suggestions
        if not sug.display:
            return
        chosen = command or sug.current()
        if chosen is None:
            return
        inp = self.query_one("#prompt-input", PromptInput)
        # Trailing space so the user can type args; it also drops the input out
        # of "typing a command" state, which hides the popup via Changed.
        inp.text = f"{chosen} "
        inp.move_cursor(inp.document.end)
        inp.focus()

    @on(PromptInput.HistoryNav)
    def _on_history_nav(self, msg: PromptInput.HistoryNav) -> None:
        # When the suggestion popup is open, Up/Down move its highlight instead
        # of browsing input history.
        if self.suggestions.display:
            self.suggestions.move(msg.delta)
            return
        if not self._history:
            return
        inp = self.query_one("#prompt-input", PromptInput)
        if msg.delta < 0:  # older
            if self._hist_idx > 0:
                self._hist_idx -= 1
                inp.text = self._history[self._hist_idx]
        elif self._hist_idx < len(self._history) - 1:  # newer
            self._hist_idx += 1
            inp.text = self._history[self._hist_idx]
        else:  # past the newest -> back to a fresh empty line
            self._hist_idx = len(self._history)
            inp.text = ""

    # --- bindings -------------------------------------------------------

    def action_interrupt_or_quit(self) -> None:
        now = time.monotonic()
        if now - self._ctrlc_ts < 1.5:
            self.exit(0)
            return
        self._ctrlc_ts = now
        # Esc is the in-flight interrupt (see action_cancel); Ctrl+C is the
        # quit affordance — double-tap within 1.5s to confirm.
        self.toast("press Ctrl+C again within 1.5s to quit")

    def action_quit_app(self) -> None:
        self.exit(0)

    def action_toggle_tool(self) -> None:
        # Expand/collapse the nearest Collapsible (tool block) up from focus.
        node: DOMNode | None = self.focused
        while node is not None:
            if isinstance(node, Collapsible):
                node.collapsed = not node.collapsed
                return
            node = node.parent

    async def action_clear_log(self) -> None:
        # Screen-only clear (Ctrl+L): wipe the transcript but keep the gateway
        # session, so the model still has context. `/clear` uses the heavier
        # _clear_and_reset to also drop that context.
        await self.transcript.remove_children()
        self._router = Router(self)  # drop active turn + tool/child registries

    async def _clear_and_reset(self) -> None:
        """`/clear`: wipe the transcript AND cold-reset the gateway session so
        the model's context is cleared and the next message opens a fresh
        session. Forwards the gateway's ``/end`` (shut down + forget mapping)."""
        await self.action_clear_log()
        await self._send(
            {
                "channel": "terminal",
                "sender_id": self._sender_id,
                "chat_id": self._chat_id,
                "content": "/end",
            }
        )

    async def action_cancel(self) -> None:
        if self.suggestions.display:
            self.suggestions.populate([])  # Esc closes the popup first
            return
        if self._in_flight:
            await self.send_interrupt()
            self.toast("interrupting…")
            return
        inp = self.query_one("#prompt-input", PromptInput)
        if inp.text.strip():
            inp.text = ""
            return
        self.toast("nothing to cancel")

    async def on_unmount(self) -> None:
        if self._consumer is not None and not self._consumer.done():
            self._consumer.cancel()


async def run_tui(
    *,
    client: TerminalClient,
    sender_id: str,
    chat_id: str,
    theme: str = "dark",
) -> int:
    app = AgentMTui(
        client=client, sender_id=sender_id, chat_id=chat_id, theme=theme
    )
    return await app.run_async() or 0


__all__ = ["AgentMTui", "run_tui"]
