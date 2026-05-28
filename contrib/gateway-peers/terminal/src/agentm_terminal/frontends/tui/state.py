"""Reactive status model for the StatusBar.

A plain dataclass the app mutates from ``usage`` / ``turn_start`` /
``session_ready`` / ``api_register`` / ``cost_budget_exceeded`` frames; the
StatusBar renders :meth:`StatusModel.line`. Kept separate from the widgets so
the status vocabulary has one home (see ``.claude/designs/textual-tui.md`` §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .theme import PHASE_GLYPH, Phase


@dataclass
class StatusModel:
    model: str = ""
    phase: Phase = "idle"
    tokens_in: int = 0
    tokens_out: int = 0
    tool_count: int = 0
    budget_exceeded: bool = False

    def line(self) -> str:
        glyph = PHASE_GLYPH.get(self.phase, "●")
        parts: list[str] = [f"{glyph} {self.phase}"]
        if self.model:
            parts.append(self.model)
        if self.tokens_in or self.tokens_out:
            parts.append(f"↑{self.tokens_in} ↓{self.tokens_out}")
        if self.tool_count:
            parts.append(f"{self.tool_count} tools")
        if self.budget_exceeded:
            parts.append("$⚠")
        return "  ·  ".join(parts)


@dataclass
class Catalog:
    """Running snapshot of what the runtime exposes, fed from session_ready /
    api_register / extension_install / cost_budget_exceeded frames. Backs the
    command palette and the /tools /extensions /budget info modals."""

    tools: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    # module_path -> last phase seen ("start"/"end"/"error[: msg]").
    extensions: dict[str, str] = field(default_factory=dict)
    budget: str | None = None  # human summary once a budget is exceeded

    def add_tool(self, name: str) -> None:
        if name and name not in self.tools:
            self.tools.append(name)

    def add_command(self, name: str) -> None:
        if name and name not in self.commands:
            self.commands.append(name)

    def tools_text(self) -> str:
        return "\n".join(self.tools) if self.tools else "(no tools registered yet)"

    def extensions_text(self) -> str:
        if not self.extensions:
            return "(no extension activity observed yet)"
        return "\n".join(f"{mod}  —  {st}" for mod, st in sorted(self.extensions.items()))

    def budget_text(self) -> str:
        return self.budget or "Budget OK (no cost_budget_exceeded seen)."


__all__ = ["Catalog", "StatusModel"]
