"""Default user-visible text for kernel-emitted ``ToolErrorEvent``s.

The kernel ``AgentLoop`` emits :class:`ToolErrorEvent` whenever a tool call
cannot produce a normal result — execution raised, the tool name is
unknown, or a ``tool_call`` handler blocked it. The kernel itself only
constructs an empty :class:`ToolResult` (``is_error=True``); writing the
human-readable English text into ``result.content`` is policy, so it lives
in this atom.

Error messages are **model-facing diagnostic text**, not developer-facing
tracebacks. Each message is phrased so the model can decide its next
action: retry with corrected args, try a different approach, or stop.

Replacing this atom (or stacking another handler ahead of it on the
``tool_error`` channel) is the supported way to localize, re-format, or
suppress those strings without patching the kernel.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import TextContent, ToolErrorEvent
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="tool_error_messages",
    description=(
        "Model-facing diagnostic text for kernel-emitted "
        "ToolErrorEvent (execution_failed / unknown_tool / blocked)."
    ),
    registers=("event:tool_error",),
    config_schema=None,
    requires=(),  # Leaf atom: formats tool_error events only.
)


def _extract_exception_summary(reason: str) -> str:
    """Extract the final exception line from a traceback string.

    A full traceback is noise for the model; the last line
    (e.g. ``FileNotFoundError: No such file: 'x.py'``) carries the
    actionable signal.
    """
    stripped = reason.strip()
    if "\n" not in stripped:
        return stripped
    last_line = stripped.rsplit("\n", 1)[-1].strip()
    return last_line if last_line else stripped


class _ToolErrorMessagesRuntime:
    def __init__(self, session: Any) -> None:
        self._session = session

    def install(self) -> None:
        self._session.bus.on(ToolErrorEvent.CHANNEL, self.on_tool_error)

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        if event.result.content:
            return
        event.result.content.append(
            TextContent(type="text", text=self._message_for(event))
        )

    def _message_for(self, event: ToolErrorEvent) -> str:
        if event.kind == "execution_failed":
            summary = _extract_exception_summary(event.reason)
            return (
                f"Tool '{event.tool_name}' raised an error: {summary}\n"
                f"Adjust your arguments or try a different approach."
            )
        if event.kind == "unknown_tool":
            return (
                f"No tool named '{event.tool_name}' is registered. "
                f"Check the tool name and try again."
            )
        if event.kind == "user_rejected":
            return (
                f"Tool '{event.tool_name}' was denied by the user. "
                f"Try a different approach instead of retrying the same call."
            )
        if event.kind == "blocked":
            return (
                f"Tool '{event.tool_name}' was blocked: {event.reason}\n"
                f"This call is not allowed under the current policy."
            )
        return f"tool_error: {event.kind} ({event.tool_name})"  # pragma: no cover


def install(session: Any, config: dict[str, Any]) -> None:
    del config
    _ToolErrorMessagesRuntime(session).install()
