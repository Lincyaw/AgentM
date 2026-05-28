"""Minimal Textual frontend for ``agentm-terminal``.

Activated via ``--ui textual``. Connects to the gateway via a
``WireClient`` (the CLI sets the wiring up; this module only renders
and dispatches input).

Scope is deliberately small — the legacy in-process
``src/agentm/modes/textual_app.py`` rendered tool-call lifecycle,
subagent blocks, thinking blocks, and the cost-budget panel by
subscribing to ``AgentSession.bus`` directly. That coupling is gone
once the session lives in a separate worker process; the only
information the chat client receives over the wire is an
``OutboundMessage`` body (``content``, ``buttons``, ``metadata``).
So this frontend renders exactly that:

* user turn (right-aligned ``> …`` line)
* assistant turn (markdown-rendered content)
* approval buttons (clickable / keyboard-pickable)
* simple status header (just the title)
* prompt input with ``Ctrl+Enter`` to send

That's it — no tool-call panels, no diagnostics, no cost. If those
matter to a deployment, ship them through ``OutboundMessage.metadata``
and add a new render path here later.

Slash commands handled locally (don't round-trip the wire):

* ``/help`` — show key bindings
* ``/quit`` — exit cleanly

Everything else is forwarded as inbound content.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.markdown import Markdown
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea

__all__ = ["run_textual"]


# Send callback signature: takes an inbound body dict, returns awaitable.
SendInbound = Callable[[dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------- widgets


class UserTurn(Static):
    """One user line in the conversation log."""

    def __init__(self, content: str) -> None:
        super().__init__(f"> {content}", markup=False)


class AssistantTextBlock(Static):
    """One assistant message rendered as markdown."""

    def __init__(self, content: str) -> None:
        super().__init__(Markdown(content))


class ApprovalBlock(Vertical):
    """A small inline block listing approval buttons.

    Click / press a key to send back the matching ``button_value``.
    Keyboard: digits 1..9 select buttons in display order.
    """

    def __init__(
        self,
        *,
        content: str,
        buttons: list[dict[str, Any]],
        on_select: Callable[[str], Awaitable[None]],
    ) -> None:
        super().__init__()
        self._buttons = buttons
        self._on_select = on_select
        self._content = content

    def compose(self) -> ComposeResult:
        if self._content:
            yield Static(self._content, classes="label")
        for idx, btn in enumerate(self._buttons, start=1):
            label = str(btn.get("label", btn.get("value", f"option {idx}")))
            value = str(btn.get("value", ""))
            yield Static(f"  [{idx}] {label}   value={value}", classes="choice")

    async def on_key(self, event: Any) -> None:
        # Map digit keys 1..9 to the corresponding button.
        key = getattr(event, "key", None)
        if not isinstance(key, str) or not key.isdigit():
            return
        idx = int(key) - 1
        if 0 <= idx < len(self._buttons):
            value = str(self._buttons[idx].get("value", ""))
            if value:
                await self._on_select(value)


class PromptInput(TextArea):
    """Bottom-bar input. ``Ctrl+Enter`` sends; ``Enter`` inserts a newline."""

    BINDINGS = [
        Binding("ctrl+j", "submit_prompt", "send", show=False, priority=True),
    ]


# ---------------------------------------------------------------- screens


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,enter,space", "dismiss(None)", "dismiss")]

    def compose(self) -> ComposeResult:
        yield Static(
            "agentm-terminal — Textual UI\n\n"
            "  Ctrl+Enter   send prompt\n"
            "  /help        show this screen\n"
            "  /quit        exit\n"
            "  digits 1..9  select approval button\n"
            "  Ctrl+C       quit (interrupt)",
            id="help-body",
        )


# ---------------------------------------------------------------- app


_CSS_PATH = Path(__file__).parent / "textual_app.tcss"


class _TerminalApp(App[int]):
    """The Textual app driver.

    The CLI constructs us with a ``send_inbound`` coroutine and an
    ``outbound_queue`` it fills from the WireClient's ``on_outbound``.
    The app drains the queue on a background task and renders each
    body dict into the conversation log.
    """

    CSS_PATH = _CSS_PATH
    BINDINGS = [
        Binding("ctrl+c", "quit_app", "quit", priority=True),
        Binding("ctrl+q", "quit_app", "quit", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        send_inbound: SendInbound,
        outbound_queue: asyncio.Queue[dict[str, Any]],
        sender_id: str,
        chat_id: str,
        title: str = "agentm-terminal",
    ) -> None:
        super().__init__()
        self._send_inbound = send_inbound
        self._outbound_queue = outbound_queue
        self._sender_id = sender_id
        self._chat_id = chat_id
        self._title = title
        self._consumer_task: asyncio.Task[Any] | None = None

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="status-header")
        yield VerticalScroll(id="conversation-log")
        yield Vertical(
            PromptInput(id="prompt-input", show_line_numbers=False),
            id="input-bar",
        )

    async def on_mount(self) -> None:
        self._consumer_task = asyncio.create_task(
            self._consume(), name="terminal-textual-consume"
        )
        # Sentinel to indicate the queue is drained / connection closed.
        # The consumer exits the app loop when it sees ``None``.

    async def _consume(self) -> None:
        while True:
            body = await self._outbound_queue.get()
            if body is None:  # type: ignore[unreachable]
                self.exit(0)
                return
            try:
                await self._render(body)
            except Exception as exc:  # noqa: BLE001
                # A bad outbound shouldn't crash the UI — log and move on.
                # ``App.log`` is a forwarding callable, not a stdlib
                # logger; use it with a single message string.
                self.log(f"render failed: body={body!r} exc={exc!r}")

    async def _render(self, body: dict[str, Any]) -> None:
        kind = str(body.get("kind") or "message")
        if kind == "turn_complete":
            return  # nothing visual — the next prompt is implicit
        log = self.query_one("#conversation-log", VerticalScroll)
        content = str(body.get("content") or "")
        buttons = body.get("buttons")
        if isinstance(buttons, list) and buttons:
            await log.mount(
                ApprovalBlock(
                    content=content,
                    buttons=[b for b in buttons if isinstance(b, dict)],
                    on_select=self._send_button_value,
                )
            )
        elif content:
            await log.mount(AssistantTextBlock(content))
        log.scroll_end(animate=False)

    async def _send_button_value(self, button_value: str) -> None:
        await self._send_inbound(
            {
                "channel": "terminal",
                "sender_id": self._sender_id,
                "chat_id": self._chat_id,
                "content": f"[button click: {button_value}]",
                "button_value": button_value,
            }
        )

    @on(TextArea.Changed)
    def _on_text_changed(self, _event: TextArea.Changed) -> None:
        # We intercept submit via the binding; let the textarea handle
        # its own editing otherwise.
        pass

    async def action_submit_prompt(self) -> None:
        inp = self.query_one("#prompt-input", PromptInput)
        text = inp.text.strip()
        if not text:
            return
        inp.text = ""
        # Local slash commands short-circuit the wire.
        if text == "/quit":
            self.exit(0)
            return
        if text == "/help":
            await self.push_screen(HelpScreen())
            return
        # Echo as user turn.
        log = self.query_one("#conversation-log", VerticalScroll)
        await log.mount(UserTurn(text))
        log.scroll_end(animate=False)
        # Send to gateway.
        try:
            await self._send_inbound(
                {
                    "channel": "terminal",
                    "sender_id": self._sender_id,
                    "chat_id": self._chat_id,
                    "content": text,
                }
            )
        except Exception:  # noqa: BLE001
            await log.mount(
                AssistantTextBlock(
                    "*(failed to send to gateway — see stderr)*"
                )
            )

    async def action_quit_app(self) -> None:
        self.exit(0)

    async def on_unmount(self) -> None:
        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()


async def run_textual(
    *,
    send_inbound: SendInbound,
    outbound_queue: asyncio.Queue[dict[str, Any]],
    sender_id: str,
    chat_id: str,
    title: str = "agentm-terminal",
) -> int:
    app = _TerminalApp(
        send_inbound=send_inbound,
        outbound_queue=outbound_queue,
        sender_id=sender_id,
        chat_id=chat_id,
        title=title,
    )
    return await app.run_async() or 0
