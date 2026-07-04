"""Policy-gate atom for the §7 ``extensions.builtin.tool_filter`` row.

Filters the final tool catalog on ``agent_start`` after tool atoms have
registered.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import AgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest


class ToolFilterConfig(BaseModel):
    model_config = {"extra": "allow"}

    allow: list[str] = []
    deny: list[str] = []


MANIFEST = ExtensionManifest(
    name="tool_filter",
    description="Remove tools from the registered catalog by allow/deny rules.",
    registers=("event:agent_start",),
    config_schema=ToolFilterConfig,
    requires=(),  # Defers filtering to agent_start so tool atoms may load in any order.
    tier=2,
)


class _ToolFilterRuntime:
    def __init__(self, api: ExtensionAPI, config: ToolFilterConfig) -> None:
        self._api = api
        self._allow = {str(name) for name in config.allow}
        self._deny = {str(name) for name in config.deny}
        self._filtered = False

    def active(self) -> bool:
        return bool(self._allow or self._deny)

    def install(self) -> None:
        self._api.on(AgentStartEvent.CHANNEL, self.on_agent_start)

    def on_agent_start(self, _: AgentStartEvent) -> None:
        if self._filtered:
            return
        tools = self._api.tools
        tool_names = {tool.name for tool in tools}
        missing_allowed = sorted(self._allow - tool_names)
        if missing_allowed:
            raise RuntimeError(
                "tool_filter allow-list names are not registered: "
                + ", ".join(missing_allowed)
            )
        tools[:] = [
            tool
            for tool in tools
            if tool.name not in self._deny
            and (not self._allow or tool.name in self._allow)
        ]
        self._filtered = True


def install(api: ExtensionAPI, config: ToolFilterConfig) -> None:
    runtime = _ToolFilterRuntime(api, config)
    if not runtime.active():
        return
    runtime.install()
