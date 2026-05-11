"""TerminalChannel — stdin/stdout adapter for local validation.

Drives the gateway from a terminal without a chat platform. Same
``BaseChannel`` contract as Feishu so what you validate here is what
runs in chat.

Two output formats, picked at config time:

* ``format: text`` (default) — human-friendly: ANSI colors, ``agent ▸``
  prefix, ``[N] Approve`` button rendering. What you want when typing
  interactively.
* ``format: json`` — one JSON object per line on stdout:

  * ``{"kind":"ready"}`` — channel started, you may start sending.
  * ``{"kind":"message","content":"…","buttons":[{...}]}`` — assistant
    text (and any approval buttons).
  * ``{"kind":"turn_complete"}`` — end of the current turn. Readers
    poll until they see this before sending the next line.
  * ``{"kind":"stopped"}`` — channel is shutting down; no further
    output on stdout.

  Logging stays on stderr regardless of format, so callers can
  ``2>/tmp/gw.err`` and parse stdout cleanly. This is the mode an
  outside script (test harness, agent driver) should use — request /
  response framing is exact and there's no ANSI noise.

Configuration::

    channels:
      terminal:
        enabled: true
        allow_from: ["*"]
        sender_id: local        # InboundMessage.sender_id (default "local")
        chat_id: terminal       # session scope (default "terminal")
        format: text             # "text" (humans) or "json" (scripts)
        color: true              # only meaningful in text mode

The ``BUTTONS`` rendering hint in text mode is ``[N] <label>`` plus
the literal ``value=…`` next to each so the user can type
``=<button_value>`` to round-trip a click. In JSON mode the full
typed button is just on the JSON object.
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from typing import Any, Literal

from ..base import BaseChannel
from ..bus import Button, OutboundKind, OutboundMessage


_RESET = "\033[0m"
_DIM = "\033[2m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"


Format = Literal["text", "json"]


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
        fmt_raw = str(cfg.get("format", "text")).lower()
        if fmt_raw not in ("text", "json"):
            raise ValueError(
                f"terminal format must be 'text' or 'json' (got {fmt_raw!r})"
            )
        self._format: Format = "json" if fmt_raw == "json" else "text"
        # Default to color when attached to a tty; explicit ``False``
        # wins for capture/CI runs. ANSI is also forced off in JSON
        # mode — JSON output is meant to be parsed.
        cfg_color = cfg.get("color")
        if self._format == "json":
            self._color = False
        elif cfg_color is None:
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
            "format": "text",
            "color": True,
        }

    async def start(self) -> None:
        if self._running:
            return
        warnings.warn(
            "TerminalChannel (in-process) is deprecated and will be removed in "
            "a future release. Run the gateway with --bind unix:///path and "
            "connect with `agentm-terminal --connect unix:///path` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._running = True
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="terminal-read"
        )
        if self._format == "json":
            self._emit({"kind": "ready"})
        else:
            self._print(
                self._style(
                    "[terminal channel ready — type /help for commands, "
                    "Ctrl-D to exit]",
                    _DIM,
                )
            )
        await self._stopped.wait()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stopped.set()
        if self._format == "json":
            self._emit({"kind": "stopped"})
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def send(self, msg: OutboundMessage) -> None:
        if self._format == "json":
            await self._send_json(msg)
            return
        await self._send_text(msg)

    # --- text rendering -----------------------------------------------

    async def _send_text(self, msg: OutboundMessage) -> None:
        if msg.kind is OutboundKind.TURN_COMPLETE:
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

    # --- json rendering -----------------------------------------------

    async def _send_json(self, msg: OutboundMessage) -> None:
        if msg.kind is OutboundKind.TURN_COMPLETE:
            self._emit({"kind": "turn_complete"})
            return
        payload: dict[str, Any] = {
            "kind": "message",
            "content": msg.content,
        }
        if msg.buttons:
            payload["buttons"] = [
                {"label": b.label, "value": b.value, "style": b.style}
                for b in msg.buttons
            ]
        # Forward channel-private metadata so callers needing the
        # ``approval_request`` / ``approval_resolved`` kind can match
        # without re-parsing button labels.
        if msg.metadata:
            payload["metadata"] = dict(msg.metadata)
        self._emit(payload)

    # --- read loop ----------------------------------------------------

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except Exception:  # pragma: no cover — defensive
                line = ""
            if not line:
                # EOF (Ctrl-D / piped input exhausted). Signal shutdown.
                self._stopped.set()
                return
            content = line.rstrip("\n")
            if not content:
                continue
            await self._dispatch_user_line(content)

    async def _dispatch_user_line(self, content: str) -> None:
        # ``=<button_value>`` is the round-trip escape for approval
        # button clicks. Mirrors what a Feishu cardAction event carries.
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
        print(text, flush=True)

    @staticmethod
    def _emit(obj: dict[str, Any]) -> None:
        """Write one JSON line, ASCII-safe (ensure_ascii=False keeps
        Chinese readable; the trailing flush makes the line visible to
        a piped reader without buffering delay)."""
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


__all__ = ["TerminalChannel"]
