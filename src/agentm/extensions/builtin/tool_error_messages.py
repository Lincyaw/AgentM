"""Default user-visible text for kernel-emitted ``ToolErrorEvent``s.

The kernel ``AgentLoop`` emits :class:`ToolErrorEvent` whenever a tool call
cannot produce a normal result — execution raised, the tool name is
unknown, or a ``tool_call`` handler blocked it. The kernel itself only
constructs an empty :class:`ToolResult` (``is_error=True``); writing the
human-readable English text into ``result.content`` is policy, so it lives
in this atom.

Replacing this atom (or stacking another handler ahead of it on the
``tool_error`` channel) is the supported way to localize, re-format, or
suppress those strings without patching the kernel. Today's exact
strings — ``"Tool execution error: {exc}"``, ``"Unknown tool: {name}"``,
``"Tool call blocked: {reason}"`` — ship as the default so users who run
with the default scenario see no behavior change from the pre-extraction
implementation.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import TextContent, ToolErrorEvent
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_error_messages",
    description=(
        "Default English user-visible text for kernel-emitted "
        "ToolErrorEvent (execution_failed / unknown_tool / blocked)."
    ),
    registers=("event:tool_error",),
    config_schema=None,
    requires=(),  # Leaf atom: formats tool_error events only.
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    def _on_tool_error(event: ToolErrorEvent) -> None:
        # Only synthesize text if no earlier handler already populated the
        # result. This makes stacking ergonomic: a localization atom that
        # runs at PRE priority can fill ``result.content`` and we silently
        # step aside.
        if event.result.content:
            return
        if event.kind == "execution_failed":
            text = f"Tool execution error: {event.reason}"
        elif event.kind == "unknown_tool":
            text = f"Unknown tool: {event.tool_name}"
        elif event.kind == "blocked":
            text = f"Tool call blocked: {event.reason}"
        else:  # pragma: no cover — defensive: Literal narrows to the three above
            text = f"tool_error: {event.kind} ({event.tool_name})"
        event.result.content.append(TextContent(type="text", text=text))

    api.on(ToolErrorEvent.CHANNEL, _on_tool_error)
