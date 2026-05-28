"""Modal screens: help, info snapshots (/tools /extensions /budget), and the
command palette (Ctrl+R). See ``.claude/designs/textual-tui.md`` §5.4."""

from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static

_HELP = """\
keys
  Enter            send                  Ctrl+J / Shift+Enter  newline
  Esc              interrupt running turn, else clear the draft
  Ctrl+C           interrupt (twice to quit)        Ctrl+D  quit
  Ctrl+L           clear transcript (visual only)   Ctrl+R  command palette
  Ctrl+E           expand/collapse a focused tool block

slash commands (typed or via the palette)
  /help  /clear  /copy-last  /tools  /extensions  /budget  /quit
  anything else is forwarded to the gateway
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,enter,space,q", "close", "close")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("▎ agentm-terminal", classes="modal-title"),
            Static(_HELP, markup=False),
            id="help-body",
        )

    def action_close(self) -> None:
        self.dismiss(None)


class InfoModal(ModalScreen[None]):
    """Title + body snapshot (/tools, /extensions, /budget)."""

    BINDINGS = [Binding("escape,enter,space,q", "close", "close")]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"▎ {self._title}", classes="modal-title"),
            Static(self._body, markup=False),
            id="info-body",
        )

    def action_close(self) -> None:
        self.dismiss(None)


class CommandPalette(ModalScreen[str | None]):
    """Filterable command picker. Dismisses with the chosen command string
    (or ``None`` on cancel)."""

    BINDINGS = [Binding("escape", "close", "cancel")]

    def __init__(self, commands: Iterable[str]) -> None:
        super().__init__()
        self._all = list(dict.fromkeys(commands))  # dedupe, preserve order

    def compose(self) -> ComposeResult:
        yield Vertical(
            Input(placeholder="filter commands…", id="palette-filter"),
            OptionList(*self._all, id="palette-list"),
            id="palette",
        )

    def on_mount(self) -> None:
        self.query_one("#palette-filter", Input).focus()

    def _matches(self, query: str) -> list[str]:
        q = query.strip().lower()
        return [c for c in self._all if q in c.lower()] if q else self._all

    def on_input_changed(self, event: Input.Changed) -> None:
        ol = self.query_one("#palette-list", OptionList)
        ol.clear_options()
        ol.add_options(self._matches(event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the filter picks the top match.
        matches = self._matches(event.value)
        self.dismiss(matches[0] if matches else None)

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        self.dismiss(str(event.option.prompt))

    def action_close(self) -> None:
        self.dismiss(None)


__all__ = ["CommandPalette", "HelpScreen", "InfoModal"]
