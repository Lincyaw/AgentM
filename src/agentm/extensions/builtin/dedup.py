"""Policy-gate atom for the §7 ``extensions.builtin.dedup`` row.

Blocks recently repeated ``(tool_name, args)`` calls within a bounded window.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import AgentStartEvent, ToolCallEvent
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


class DedupConfig(BaseModel):
    window: int = 10


MANIFEST = ExtensionManifest(
    name="dedup",
    description="Block recently repeated tool calls within a sliding window.",
    registers=("event:agent_start", "event:tool_call"),
    config_schema=DedupConfig,
    requires=(),  # Leaf atom: observes tool calls without requiring tool atoms.
)


def _make_key(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
    return (tool_name, json.dumps(args, sort_keys=True, separators=(",", ":"), default=str))


def install(api: ExtensionAPI, config: DedupConfig) -> None:
    window = max(0, config.window)
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
