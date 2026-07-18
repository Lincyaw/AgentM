"""Policy-gate atom for the §7 ``extensions.builtin.tool_filter`` row.

Filters the final tool catalog on ``before_agent_start`` (with an
``agent_start`` safety net), after tool atoms have registered. Filtering
early lets prompt-building atoms registered after this one (e.g.
``tool_index``) observe the final catalog.
"""

from __future__ import annotations
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import BeforeRunEvent
from agentm.extensions import ChannelEffects, ExtensionManifest


class ToolFilterConfig(BaseModel):
    model_config = {"extra": "allow"}

    allow: list[str] = []
    deny: list[str] = []


MANIFEST = ExtensionManifest(
    name="tool_filter",
    description="Remove tools from the registered catalog by allow/deny rules.",
    registers=("event:before_agent_start", "event:agent_start"),
    config_schema=ToolFilterConfig,
    requires=(),  # Defers filtering to start events so tool atoms may load in any order.
    tier=2,
    effects={
        "before_agent_start": ChannelEffects(mutates=("tools",)),
        "agent_start": ChannelEffects(mutates=("tools",)),
    },
)


class _ToolFilterRuntime:
    def __init__(self, session: Any, config: ToolFilterConfig) -> None:
        self._session = session
        self._allow = {str(name) for name in config.allow}
        self._deny = {str(name) for name in config.deny}
        self._filtered = False

    def active(self) -> bool:
        return bool(self._allow or self._deny)

    def install(self) -> None:
        # Filter on before_agent_start so prompt-building atoms registered
        # after this one (e.g. tool_index) see the final catalog; keep the
        # agent_start hook as a safety net for sessions that skip the
        # before event. Filtering is idempotent.
        self._session.bus.on(BeforeRunEvent.CHANNEL, self.on_agent_start)
        self._session.bus.on(BeforeRunEvent.CHANNEL, self.on_agent_start)

    def on_agent_start(self, _: BeforeRunEvent | BeforeRunEvent) -> None:
        if self._filtered:
            return
        tools = self._session.tools
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


def install(session: Any, config: ToolFilterConfig) -> None:
    runtime = _ToolFilterRuntime(session, config)
    if not runtime.active():
        return
    runtime.install()
