"""Rendering helpers for the v2 ``outbound`` envelope body (§2.5).

The wire carries outbound payloads as plain dicts. The discriminator is
``metadata.kind`` (one of ``assistant_text`` / ``approval_request`` /
``approval_resolved`` / ``diagnostic_warning`` / ``diagnostic_error``).

* text mode: ``agent ▸`` prefix; diagnostics get a ``⚠``/``✖`` prefix;
  ANSI-coloured ``[N] label   value=…`` button block with the
  ``(type =<value> to round-trip a button click)`` hint.
* json mode: ``{"kind":"ready"}`` after handshake, then one
  ``{"kind":"message",…}`` per outbound (carrying ``content``, optional
  ``buttons``, optional ``metadata``), ``{"kind":"stopped"}`` on shutdown.
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


class Renderer:
    """Stateless-ish renderer wrapping stdout writes.

    ``color`` is honoured in text mode only; JSON mode is always
    plain ASCII / UTF-8 so a downstream parser sees clean lines.
    """

    def __init__(self, *, fmt: str, color: bool) -> None:
        if fmt not in ("text", "json"):
            raise ValueError(
                f"renderer format must be 'text' or 'json' (got {fmt!r})"
            )
        self._format = fmt
        # JSON mode forces ANSI off (the legacy channel did the same).
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
        """Render one v2 ``outbound`` envelope body (§2.5).

        Fields read:

        * ``content`` — assistant text / card body (may be empty).
        * ``buttons`` — optional list of ``{"label","value","style"}``.
        * ``metadata.kind`` — render-style discriminator.
        """
        raw_meta = body.get("metadata")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        meta_kind = str(meta.get("kind") or "assistant_text")
        if self._format == "json":
            self._render_json(meta_kind, body)
        else:
            self._render_text(meta_kind, body)

    # --- text ----------------------------------------------------------

    def _render_text(self, meta_kind: str, body: dict[str, Any]) -> None:
        content = str(body.get("content") or "")
        text = content.rstrip()
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
                self._style(
                    "  (type =<value> to round-trip a button click)", _DIM
                )
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
        if metadata:
            payload["metadata"] = dict(metadata)
        else:
            payload["metadata"] = {"kind": meta_kind}
        self._emit(payload)

    # --- low-level -----------------------------------------------------

    def _style(self, text: str, code: str) -> str:
        if not self._color:
            return text
        return f"{code}{text}{_RESET}"

    @staticmethod
    def _print(text: str) -> None:
        print(text, flush=True)

    @staticmethod
    def _emit(obj: dict[str, Any]) -> None:
        """One JSON object per line on stdout. ``ensure_ascii=False``
        keeps non-ASCII (e.g. Chinese) human-readable."""
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


__all__ = ["Renderer"]
