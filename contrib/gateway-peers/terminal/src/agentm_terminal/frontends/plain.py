"""Non-TTY renderer for the v2 ``outbound`` body (text / json).

The frontend used when stdout is not a TTY (scripts, agent drivers) or when
``--format text|json`` is passed. Behaviour is unchanged from the previous
``renderer.py``: text mode prints ``agent ▸``-prefixed lines + an ANSI button
block; json mode emits one ``{"kind": ...}`` object per line. The rich TUI
(``frontends.tui``) is the default on a TTY.

Discriminator is ``metadata.kind`` (assistant_text / approval_request /
approval_resolved / diagnostic_warning / diagnostic_error); the streaming /
control kinds the gateway now also emits are rendered as plain assistant text
or ignored here — the TUI is the surface that renders them richly.
"""

from __future__ import annotations

import json
import sys
from typing import Any

_RESET = "\033[0m"
_DIM = "\033[2m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"

# Streaming/lifecycle kinds the plain renderer deliberately drops (they are
# live decoration for the TUI; a piped consumer wants whole messages).
_PLAIN_SKIP_KINDS = frozenset(
    {
        "turn_start",
        "stream_text",
        "stream_thinking",
        "tool_call",
        "tool_result",
        "usage",
        "child_start",
        "child_end",
        "agent_end",
        "extension_install",
        "extension_reload",
        "extension_unload",
        "api_register",
        "api_send_user_message",
        "resource_write",
        "plan_submitted",
        "after_compact",
        "cost_budget_exceeded",
        "session_ready",
        "command_dispatched",
    }
)


class PlainRenderer:
    """Stateless-ish renderer wrapping stdout writes."""

    def __init__(self, *, fmt: str, color: bool) -> None:
        if fmt not in ("text", "json"):
            raise ValueError(f"renderer format must be 'text' or 'json' (got {fmt!r})")
        self._format = fmt
        self._color = False if fmt == "json" else color

    # --- lifecycle markers ---------------------------------------------

    def ready(self) -> None:
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

    def stopped(self) -> None:
        if self._format == "json":
            self._emit({"kind": "stopped"})

    # --- per-outbound --------------------------------------------------

    def render_outbound(self, body: dict[str, Any]) -> None:
        raw_meta = body.get("metadata")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        meta_kind = str(meta.get("kind") or "assistant_text")
        if meta_kind in _PLAIN_SKIP_KINDS:
            return
        if self._format == "json":
            self._render_json(meta_kind, body)
        else:
            self._render_text(meta_kind, body)

    # --- text ----------------------------------------------------------

    def _render_text(self, meta_kind: str, body: dict[str, Any]) -> None:
        text = str(body.get("content") or "").rstrip()
        if text:
            if meta_kind == "diagnostic_error":
                self._print(self._style("✖ " + text, _YELLOW))
            elif meta_kind == "diagnostic_warning":
                self._print(self._style("⚠ " + text, _YELLOW))
            else:
                self._print(self._style("agent ▸ ", _BLUE) + text)
        buttons = body.get("buttons") or []
        if buttons:
            for idx, btn in enumerate(buttons, start=1):
                style = str(btn.get("style") or "primary")
                label = str(btn.get("label") or "")
                value = str(btn.get("value") or "")
                color = _YELLOW if style == "danger" else _GREEN
                self._print(
                    self._style(f"  [{idx}] {label}", color)
                    + self._style(f"   value={value}", _DIM)
                )
            self._print(
                self._style("  (type =<value> to round-trip a button click)", _DIM)
            )

    # --- json ----------------------------------------------------------

    def _render_json(self, meta_kind: str, body: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "kind": "message",
            "content": str(body.get("content") or ""),
        }
        buttons = body.get("buttons")
        if buttons:
            payload["buttons"] = [
                {
                    "label": str(b.get("label") or ""),
                    "value": str(b.get("value") or ""),
                    "style": str(b.get("style") or "primary"),
                }
                for b in buttons
            ]
        metadata = body.get("metadata")
        payload["metadata"] = dict(metadata) if metadata else {"kind": meta_kind}
        self._emit(payload)

    # --- low-level -----------------------------------------------------

    def _style(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self._color else text

    @staticmethod
    def _print(text: str) -> None:
        print(text, flush=True)

    @staticmethod
    def _emit(obj: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


__all__ = ["PlainRenderer"]
