"""Design language for the terminal TUI — the visual vocabulary.

Colour lives in ``app.tcss`` (via Textual theme CSS variables); this module
owns the *symbols* and the phase state machine so every widget pulls them from
one place. See ``.claude/designs/textual-tui.md`` §5.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

CSS_PATH: Final[Path] = Path(__file__).with_name("app.tcss")

# The status phase shown in the StatusBar. Drives the leading glyph.
Phase = Literal["idle", "thinking", "streaming", "tool", "subagent"]

PHASE_GLYPH: Final[dict[str, str]] = {
    "idle": "●",
    "thinking": "◐",
    "streaming": "◑",
    "tool": "⚙",
    "subagent": "⌥",
}

# Tool-call lifecycle glyphs (rendered Rich-side inside the tool block title).
TOOL_RUNNING: Final[str] = "⟳"
TOOL_OK: Final[str] = "✓"
TOOL_ERROR: Final[str] = "✗"

# Attribution labels (the "● assistant" style prefix per turn).
LABEL_ASSISTANT: Final[str] = "● assistant"
LABEL_SYSTEM: Final[str] = "system → you"

_THEME_ALIASES: Final[dict[str, str]] = {
    "dark": "textual-dark",
    "light": "textual-light",
}


def resolve_theme(alias: str) -> str:
    """Map a CLI ``--theme`` alias (``dark`` / ``light``) to a Textual built-in
    theme name. Unknown values pass through unchanged so a user can name any
    registered Textual theme; Textual validates and the app surfaces failures.
    """
    return _THEME_ALIASES.get(alias, alias)


__all__ = [
    "CSS_PATH",
    "LABEL_ASSISTANT",
    "LABEL_SYSTEM",
    "PHASE_GLYPH",
    "Phase",
    "TOOL_ERROR",
    "TOOL_OK",
    "TOOL_RUNNING",
    "resolve_theme",
]
