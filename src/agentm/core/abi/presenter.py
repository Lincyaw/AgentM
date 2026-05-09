"""Presenter-facing view constants exported by the kernel.

The CLI and TUI render an evolving "phase" (idle / thinking / streaming /
tool / subagent) alongside the model name and turn counter. Earlier
revisions hardcoded the glyph table in ``modes/textual_app.py``, which
meant every presenter that wanted parity (a textual app, a curses wrapper,
a JSON status line for an HTTP gateway) had to copy-paste the same five
strings. The framework's posture is "kernel exports the contract,
presenters consume it" — so the canonical phase set lives here.

This module is pure data: no imports beyond ``typing``. It is part of
``core.abi``'s public surface and follows the same stability promise as
the event taxonomy. Adding a phase is a constitution change.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal, Mapping

Phase = Literal["idle", "thinking", "streaming", "tool", "subagent"]

# The default UTF-8 glyph table. Wrapped in ``MappingProxyType`` so callers
# cannot mutate the canonical surface at runtime; presenters that want a
# different look should compose a private dict, not patch this one.
PHASE_GLYPHS: Final[Mapping[Phase, str]] = MappingProxyType(
    {
        "idle": "●",        # ●
        "thinking": "◐",    # ◐
        "streaming": "◑",   # ◑
        "tool": "▶",        # ▶
        "subagent": "↳",    # ↳
    }
)

__all__ = ["Phase", "PHASE_GLYPHS"]
