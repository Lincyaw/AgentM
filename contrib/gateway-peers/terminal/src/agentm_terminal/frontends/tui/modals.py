"""The command palette (Ctrl+R). See ``.claude/designs/textual-tui.md`` §5.4."""

from __future__ import annotations

from collections.abc import Iterable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList


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


__all__ = ["CommandPalette"]
