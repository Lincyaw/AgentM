"""Policy-gate atom for the §7 ``extensions.builtin.dedup`` row.

Blocks recently repeated ``(tool_name, args)`` calls within a bounded window.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import AgentStartEvent, ExtensionAPI, ToolCallEvent
from agentm.extensions import ExtensionManifest

_ToolCallKey = tuple[str, str]


class DedupConfig(BaseModel):
    window: int = 10


MANIFEST = ExtensionManifest(
    name="dedup",
    description="Block recently repeated tool calls within a sliding window.",
    registers=("event:agent_start", "event:tool_call"),
    config_schema=DedupConfig,
    requires=(),  # Leaf atom: observes tool calls without requiring tool atoms.
)


def _make_key(tool_name: str, args: dict[str, Any]) -> _ToolCallKey:
    return (
        tool_name,
        json.dumps(args, sort_keys=True, separators=(",", ":"), default=str),
    )


class _DedupRuntime:
    def __init__(self, api: ExtensionAPI, config: DedupConfig) -> None:
        self._api = api
        self._window = max(0, config.window)
        self._recent: deque[_ToolCallKey] = deque(maxlen=self._window)

    def active(self) -> bool:
        return self._window > 0

    def install(self) -> None:
        self._api.on(AgentStartEvent.CHANNEL, self.reset)
        self._api.on(ToolCallEvent.CHANNEL, self.on_tool_call)

    def reset(self, _: AgentStartEvent) -> None:
        self._recent.clear()

    def on_tool_call(self, event: ToolCallEvent) -> dict[str, Any] | None:
        key = _make_key(event.tool_name, event.args)
        if key in self._recent:
            return {
                "block": True,
                "reason": "duplicate of recent call",
            }
        self._recent.append(key)
        return None


def install(api: ExtensionAPI, config: DedupConfig) -> None:
    runtime = _DedupRuntime(api, config)
    if not runtime.active():
        return
    runtime.install()
