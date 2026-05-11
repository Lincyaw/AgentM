"""TerminalChannel — stdin/stdout adapter for local validation.

Lets you drive the gateway from a terminal without a chat platform:
every typed line becomes an :class:`InboundMessage`, every outbound
message is printed back. Useful for trying slash commands, skill
activation, or approval flows against a real agent before pointing
Feishu at the same gateway.

This is **not** intended for production use — there is no multi-user
support, no rich rendering, and stdin reading is line-buffered. It is
a deliberate analog to nanobot's terminal mode and to the existing
``agentm`` CLI's REPL, but routed through the same channel + command
+ approval plumbing that Feishu uses, so what you validate in the
terminal is what runs in chat.

Configuration::

    channels:
      terminal:
        enabled: true
        allow_from: ["*"]
        sender_id: local        # what InboundMessage.sender_id reports (default: "local")
        chat_id: terminal       # session scope (default: "terminal")
        color: true             # ANSI colors on outbound (default: True; auto-off if stdout is not a tty)

The ``BUTTONS`` rendering hint is plain text: each :class:`Button`
becomes ``[N] <label>`` and the user types ``/approve`` or
``/deny`` (when that command lands) or pastes the literal
``button_value`` prefixed with ``=`` for the round-trip. The escape
hatch keeps the channel usable for approval flows even before the
text-fallback commands ship.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from ..base import BaseChannel
from ..bus import Button, OutboundKind, OutboundMessage


_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"


class TerminalChannel(BaseChannel):
    name = "terminal"
    display_name = "Terminal (stdin/stdout)"

    def __init__(self, config: Any, bus: Any) -> None:
        super().__init__(config, bus)
        self._stopped = asyncio.Event()
        self._reader_task: asyncio.Task[Any] | None = None
        self._last_buttons: list[Button] = []
        cfg = config if isinstance(config, dict) else {}
        self._sender_id: str = str(cfg.get("sender_id") or "local")
        self._chat_id: str = str(cfg.get("chat_id") or "terminal")
        # Default to color when attached to a tty; explicit ``False``
        # wins for capture/CI runs.
        cfg_color = cfg.get("color")
        if cfg_color is None:
            self._color = sys.stdout.isatty()
        else:
            self._color = bool(cfg_color)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "enabled": False,
            "allow_from": ["*"],
            "sender_id": "local",
            "chat_id": "terminal",
            "color": True,
        }

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="terminal-read"
        )
        self._print(
            self._style(
                "[terminal channel ready — type /help for commands, Ctrl-D to exit]",
                _DIM,
            )
        )
        await self._stopped.wait()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stopped.set()
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def send(self, msg: OutboundMessage) -> None:
        if msg.kind is OutboundKind.TURN_COMPLETE:
            # Mark end-of-turn for the user; no payload to render.
            self._print(self._style("… (turn complete)", _DIM))
            return
        text = msg.content.rstrip()
        if text:
            prefix = self._style("agent ▸ ", _BLUE)
            self._print(prefix + text)
        if msg.buttons:
            self._last_buttons = list(msg.buttons)
            for idx, btn in enumerate(msg.buttons, start=1):
                color = _YELLOW if btn.style == "danger" else _GREEN
                self._print(
                    self._style(f"  [{idx}] {btn.label}", color)
                    + self._style(f"   value={btn.value}", _DIM)
                )
            self._print(
                self._style(
                    "  (type =<value> to round-trip a button click)", _DIM
                )
            )

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except Exception:  # pragma: no cover — defensive
                line = ""
            if not line:
                # EOF (Ctrl-D / piped input exhausted). Signal shutdown
                # by setting the stopped event; the manager's start_all
                # noticing won't help because terminal.start() is what
                # holds the manager alive — emit a synthetic /end to
                # close the session cleanly, then stop.
                self._stopped.set()
                return
            content = line.rstrip("\n")
            if not content:
                continue
            await self._dispatch_user_line(content)

    async def _dispatch_user_line(self, content: str) -> None:
        # ``=<button_value>`` is the round-trip escape for approval
        # button clicks. Mirrors what a Feishu cardAction event looks
        # like to the bridge: same ``button_value``, no user prose.
        if content.startswith("="):
            button_value = content[1:].strip()
            await self._handle_message(
                sender_id=self._sender_id,
                chat_id=self._chat_id,
                content=f"[button click: {button_value}]",
                button_value=button_value or None,
            )
            return
        await self._handle_message(
            sender_id=self._sender_id,
            chat_id=self._chat_id,
            content=content,
        )

    # --- rendering helpers --------------------------------------------

    def _style(self, text: str, code: str) -> str:
        if not self._color:
            return text
        return f"{code}{text}{_RESET}"

    @staticmethod
    def _print(text: str) -> None:
        # Flush so prompts arrive before the stdin readline blocks the
        # main thread's view of the terminal.
        print(text, flush=True)


# Tag for the structural Protocol check the registry runs at startup.
__all__ = ["TerminalChannel"]
