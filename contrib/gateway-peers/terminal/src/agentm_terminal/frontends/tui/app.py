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
from textual.containers import VerticalScroll
from textual.widgets import Footer

from ...client import TerminalClient
from .router import Router
from .state import StatusModel
from .theme import CSS_PATH, Phase, resolve_theme
from .widgets import PromptInput, StatusBar, Toast, UserTurn


class AgentMTui(App[int]):
    CSS_PATH = str(CSS_PATH)
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "interrupt", priority=True, show=True),
        Binding("ctrl+d", "quit_app", "quit", priority=True, show=True),
        Binding("ctrl+l", "clear_log", "clear", show=True),
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
        self._router = Router(self)
        self._consumer: asyncio.Task[None] | None = None
        self._history: list[str] = []
        self._ctrlc_ts = 0.0
        # True from prompt-submit until the loop's agent_end; gates Esc-interrupt.
        self._in_flight = False

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        yield VerticalScroll(id="transcript")
        yield PromptInput(id="prompt-input", show_line_numbers=False)
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self.theme = self._theme_name
        except Exception:  # noqa: BLE001 — invalid theme name; keep the default
            pass
        self.show_status()
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
        if turn is not None:
            await turn.flush()
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
        if text in ("/quit", "/exit", "/q"):
            self.exit(0)
            return
        self._history.append(text)
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

    async def _send(self, body: dict[str, object]) -> None:
        try:
            await self._client.send_inbound(body)
        except Exception:  # noqa: BLE001
            self.toast("failed to send to gateway", variant="warn")

    # --- bindings -------------------------------------------------------

    def action_interrupt_or_quit(self) -> None:
        now = time.monotonic()
        if now - self._ctrlc_ts < 1.5:
            self.exit(0)
            return
        self._ctrlc_ts = now
        # Interrupting an in-flight turn needs a gateway cancel affordance
        # (Phase 3). Until then Ctrl+C only escalates to quit on double-tap.
        self.toast("press Ctrl+C again within 1.5s to quit")

    def action_quit_app(self) -> None:
        self.exit(0)

    async def action_clear_log(self) -> None:
        await self.transcript.remove_children()
        self._router = Router(self)  # drop active turn + tool/child registries

    async def action_cancel(self) -> None:
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
