"""Policy-gate atom for the §7 ``extensions.builtin.dedup`` row.

Blocks recently repeated ``(tool_name, args)`` calls within a bounded window.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from agentm.core.abi import AgentStartEvent, ToolCallEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="dedup",
    description="Block recently repeated tool calls within a sliding window.",
    registers=("event:agent_start", "event:tool_call"),
    config_schema={
        "type": "object",
        "properties": {
            "window": {"type": "integer", "minimum": 0, "default": 10},
        },
        "additionalProperties": True,
    },
    requires=(),  # Leaf atom: observes tool calls without requiring tool atoms.
)


def _make_key(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
    return (tool_name, json.dumps(args, sort_keys=True, separators=(",", ":"), default=str))


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    window = max(0, int(config.get("window", 10)))
    if window == 0:
        return

    recent: deque[tuple[str, str]] = deque(maxlen=window)

    def _reset(_: AgentStartEvent) -> None:
        recent.clear()

    def _on_tool_call(event: ToolCallEvent) -> dict[str, Any] | None:
        key = _make_key(event.tool_name, event.args)
        if key in recent:
            return {
                "block": True,
                "reason": "duplicate of recent call",
            }
        recent.append(key)
        return None

    api.on(AgentStartEvent.CHANNEL, _reset)
    api.on(ToolCallEvent.CHANNEL, _on_tool_call)
