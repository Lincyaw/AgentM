"""The small components — one widget per conversation/control concept.

Each widget owns its own render; the router (router.py) is the only place wire
shapes meet these widgets. Colour/gutters come from ``app.tcss``; glyphs/labels
come from ``theme.py``. See ``.claude/designs/textual-tui.md`` §5.2.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from rich.markdown import Markdown
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Collapsible, Static, TextArea

from .state import StatusModel
from .theme import (
    LABEL_ASSISTANT,
    LABEL_SYSTEM,
    TOOL_ERROR,
    TOOL_OK,
    TOOL_RUNNING,
)

SelectCb = Callable[[str], Awaitable[None]]


class StatusBar(Static):
    """Top dock line: phase glyph · model · tokens · counters · budget."""

    def show(self, model: StatusModel) -> None:
        self.update(model.line())


class UserTurn(Static):
    """One user line (right-of-gutter)."""

    def __init__(self, content: str) -> None:
        super().__init__(content, markup=False, classes="turn turn--user")


class SystemTurn(Vertical):
    """An extension-injected user message — visibly distinct from a typed one."""

    def __init__(self, content: str, *, source: str) -> None:
        super().__init__(classes="turn turn--system")
        self._content = content
        self._source = source

    def compose(self) -> ComposeResult:
        yield Static(f"{LABEL_SYSTEM}  ({self._source})", classes="attrib--system")
        yield Static(self._content, markup=False)


class ThinkingBlock(Collapsible):
    """Dim, collapsed-by-default reasoning text."""

    def __init__(self) -> None:
        self._body = Static("", classes="thinking", markup=False)
        super().__init__(self._body, title="thinking", collapsed=True)

    def set_text(self, text: str) -> None:
        self._body.update(text)


class AssistantText(Static):
    """Streaming markdown target for one assistant turn."""

    def set_markdown(self, text: str) -> None:
        self.update(Markdown(text) if text.strip() else "")


class AssistantTurn(Vertical):
    """An assistant turn: attribution label + (thinking) + streamed text +
    inline tool / sub-agent blocks. Owns its stream buffers; the app's flush
    timer calls :meth:`flush` at ~20 Hz so we never repaint per token."""

    def __init__(self) -> None:
        super().__init__(classes="turn turn--assistant")
        self._text_buf = ""
        self._think_buf = ""
        self._dirty_text = False
        self._dirty_think = False
        self._text_widget: AssistantText | None = None
        self._think_widget: ThinkingBlock | None = None

    def compose(self) -> ComposeResult:
        yield Static(LABEL_ASSISTANT, classes="attrib")

    def append_text(self, delta: str) -> None:
        self._text_buf += delta
        self._dirty_text = True

    def append_thinking(self, delta: str) -> None:
        self._think_buf += delta
        self._dirty_think = True

    async def flush(self) -> None:
        if self._dirty_think:
            if self._think_widget is None:
                self._think_widget = ThinkingBlock()
                await self.mount(self._think_widget)
            self._think_widget.set_text(self._think_buf)
            self._dirty_think = False
        if self._dirty_text:
            if self._text_widget is None:
                self._text_widget = AssistantText()
                await self.mount(self._text_widget)
            self._text_widget.set_markdown(self._text_buf)
            self._dirty_text = False

    async def set_final_text(self, text: str) -> None:
        # The durable turn_end text is authoritative — it replaces whatever the
        # (possibly lossy) ephemeral stream assembled.
        self._text_buf = text
        self._dirty_text = True
        await self.flush()

    async def add_block(self, widget: Any) -> None:
        await self.mount(widget)


class ToolBlock(Collapsible):
    """One tool call: title ``name ⟳/✓/✗`` + collapsible args/result body."""

    def __init__(self, *, name: str, args: dict[str, Any]) -> None:
        self._name = name
        self._args_text = _format_args(args)
        self._body = Static(self._args_text, classes="tool-body", markup=False)
        super().__init__(
            self._body, title=f"{name}  {TOOL_RUNNING}", collapsed=True
        )

    def set_result(self, *, ok: bool, content: str) -> None:
        glyph = TOOL_OK if ok else TOOL_ERROR
        self.title = f"{self._name}  {glyph}"
        body = f"{self._args_text}\n\n{content}" if content else self._args_text
        self._body.update(body)
        if not ok:
            self.collapsed = False  # surface failures


class SubagentBlock(Static):
    """A sub-agent dispatch marker. The child session has no wire_driver, so we
    show its lifecycle (purpose + running/done/error), not its inner stream."""

    def __init__(self, *, purpose: str) -> None:
        super().__init__(f"⌥ subagent: {purpose}  {TOOL_RUNNING}", markup=False)
        self._purpose = purpose

    def finish(self, *, error: str | None) -> None:
        glyph = TOOL_ERROR if error else TOOL_OK
        suffix = f"  ({error})" if error else ""
        self.update(f"⌥ subagent: {self._purpose}  {glyph}{suffix}")


class ApprovalBlock(Vertical):
    """Inline approval card: content + clickable / digit-pickable choices."""

    def __init__(
        self,
        *,
        content: str,
        buttons: list[dict[str, Any]],
        on_select: SelectCb,
    ) -> None:
        super().__init__()
        self._content = content
        self._buttons = buttons
        self._on_select = on_select

    def compose(self) -> ComposeResult:
        if self._content:
            yield Static(self._content, markup=False)
        for idx, btn in enumerate(self._buttons, start=1):
            label = str(btn.get("label", btn.get("value", f"option {idx}")))
            yield Static(f"  [{idx}] {label}", classes="approval-choice", markup=False)

    async def on_key(self, event: Any) -> None:
        key = getattr(event, "key", None)
        if not isinstance(key, str) or not key.isdigit():
            return
        idx = int(key) - 1
        if 0 <= idx < len(self._buttons):
            value = str(self._buttons[idx].get("value", ""))
            if value:
                await self._on_select(value)


class Toast(Static):
    """Transient overlay for control / observability events. Auto-dismisses."""

    def __init__(self, text: str, *, variant: str = "info", ttl: float = 4.0) -> None:
        super().__init__(text, markup=False)
        self._ttl = ttl
        if variant == "warn":
            self.add_class("-warn")
        elif variant == "selfmod":
            self.add_class("-selfmod")

    def on_mount(self) -> None:
        self.set_timer(self._ttl, self.remove)


class PromptInput(TextArea):
    """Bottom input. Enter submits; Shift+Enter (or Ctrl+J) inserts a newline."""

    BINDINGS = [
        Binding("enter", "submit", "send", show=False, priority=True),
        Binding("ctrl+j", "newline", "newline", show=False, priority=True),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter on a non-empty input."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def action_submit(self) -> None:
        # Bubble up; the App owns the send (it holds the wire client).
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")


def _format_args(args: dict[str, Any], limit: int = 600) -> str:
    if not args:
        return ""
    try:
        import json

        s = json.dumps(args, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        s = str(args)
    return s[:limit]


__all__ = [
    "ApprovalBlock",
    "AssistantText",
    "AssistantTurn",
    "PromptInput",
    "StatusBar",
    "SubagentBlock",
    "SystemTurn",
    "ThinkingBlock",
    "Toast",
    "ToolBlock",
    "UserTurn",
]
