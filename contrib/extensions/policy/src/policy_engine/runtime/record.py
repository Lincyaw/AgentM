# code-health: ignore-file[AM025] -- rendering reads untyped payloads at the
# event/trajectory boundary; the isinstance checks here are that boundary.
"""The session record: turns rendered as events, once, for every reader.

Four things read the same rendering — the tagger's batches, the critic's
prompts, fork restore, and the offline tag command. One record type and one
renderer serve all four, so a reader that drifts has to edit the same lines as
everyone else.

Turns are rendered as events, not content: which file was read, which was
edited and by how much, which command ran and what it exited with. Bodies —
file contents, diffs, stdout — are what made the input large, and none of them
decide a predicate or a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Bodies are dropped, but a command line is itself the evidence for several
# predicates ("does every test command carry a narrowing selector"), so it is
# kept whole up to a generous bound.
_COMMAND_LIMIT = 400
_RESULT_LIMIT = 200
_TEXT_LIMIT = 600
_TASK_LIMIT = 2000

#: How much of a tool result travels with its call record.
RESULT_TEXT_LIMIT = 1500


def content_text(obj: object, limit: int = 3000) -> str:
    """Extract joined .text from content blocks on any message-like object."""
    content = getattr(obj, "content", None)  # code-health: ignore[AM021]
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)  # code-health: ignore[AM021]
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)[:limit]


@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    """One tool call as the record keeps it. Satisfies ``revision.ToolCall``."""

    name: str
    arguments: Mapping[str, object]
    result_text: str
    is_error: bool

    @staticmethod
    def from_result(
        name: str, arguments: Mapping[str, object], result: object
    ) -> ToolCallRecord:
        """The one place a tool result becomes a record — live and restored
        turns must not diverge on how ``is_error`` or the text cap is read."""
        is_error = getattr(result, "is_error", False)  # code-health: ignore[AM021]
        return ToolCallRecord(
            name=name,
            arguments=dict(arguments),
            result_text=content_text(result, RESULT_TEXT_LIMIT),
            is_error=bool(is_error),
        )

    def view(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "result_text": self.result_text,
            "is_error": self.is_error,
        }


def _edit_extent(arguments: Mapping[str, object]) -> str:
    """Rough size of an edit, without carrying the edited text itself."""
    added = arguments.get("new_string")
    removed = arguments.get("old_string")
    added_lines = len(added.splitlines()) if isinstance(added, str) else 0
    removed_lines = len(removed.splitlines()) if isinstance(removed, str) else 0
    if not added_lines and not removed_lines:
        return ""
    return f" (+{added_lines}/-{removed_lines})"


def _render_tool_call(call: Mapping[str, object]) -> str:
    name = str(call.get("name", "?"))
    arguments = call.get("arguments")
    arguments = (
        arguments if isinstance(arguments, Mapping) else {}
    )  # code-health: ignore[AM025]
    is_error = bool(call.get("is_error"))
    result_text = str(call.get("result_text", ""))

    if name == "bash":
        command = str(arguments.get("cmd", ""))[:_COMMAND_LIMIT]
        status = "FAILED" if is_error else "ok"
        tail = result_text.strip().splitlines()
        detail = tail[-1][:_RESULT_LIMIT] if is_error and tail else ""
        line = f"bash {command} -> {status}"
        return f"{line}\n    {detail}" if detail else line

    path = arguments.get("path") or arguments.get("file_path")
    path = str(path) if isinstance(path, str) else ""
    suffix = _edit_extent(arguments) if name in {"edit", "write"} else ""
    status = " FAILED" if is_error else ""
    purpose = arguments.get("purpose")
    note = f"  [{str(purpose)[:80]}]" if isinstance(purpose, str) and purpose else ""
    return f"{name} {path}{suffix}{status}{note}"


def render_turn(
    turn_index: int,
    assistant_text: str,
    tool_calls: Sequence[Mapping[str, object]],
    *,
    task_text: str = "",
) -> str:
    """One turn as an event line or two — no file bodies, no diffs, no stdout."""
    parts: list[str] = []
    if task_text:
        parts.append(f"Task:\n{task_text[:_TASK_LIMIT]}\n")
    parts.append(f"Turn {turn_index}:")
    if assistant_text:
        parts.append(f"  says: {assistant_text[:_TEXT_LIMIT]}")
    for call in tool_calls:
        parts.append(f"  {_render_tool_call(call)}")
    return "\n".join(parts)


__all__ = [
    "RESULT_TEXT_LIMIT",
    "ToolCallRecord",
    "content_text",
    "render_turn",
]
