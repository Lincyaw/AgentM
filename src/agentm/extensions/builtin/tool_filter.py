"""Filter the tool set on ``before_send`` by allow/deny rules.

Every LLM call sees only the permitted catalog, regardless of the order
tool-registering atoms installed in.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority, BeforeSendEvent, Tool
from agentm.extensions import ExtensionManifest


class ToolFilterConfig(BaseModel):
    allow: list[str] = []
    deny: list[str] = []


MANIFEST = ExtensionManifest(
    name="tool_filter",
    description="Remove tools from the catalog by allow/deny rules.",
    registers=("event:before_send",),
    config_schema=ToolFilterConfig,
    requires=(),
    priority=AtomInstallPriority.POLICY,
)


class _ToolFilterRuntime:
    def __init__(self, allow: set[str], deny: set[str]) -> None:
        self._allow = allow
        self._deny = deny
        self._checked = False

    def on_before_send(self, event: BeforeSendEvent) -> dict[str, list[Tool]] | None:
        tools = event.tools
        if not self._checked:
            tool_names = {tool.name for tool in tools}
            missing = sorted(self._allow - tool_names)
            if missing:
                raise RuntimeError(
                    "tool_filter allow-list names are not registered: "
                    + ", ".join(missing)
                )
            self._checked = True
        filtered = [
            tool
            for tool in tools
            if tool.name not in self._deny
            and (not self._allow or tool.name in self._allow)
        ]
        return {"tools": filtered}


def install(api: AtomAPI, config: ToolFilterConfig) -> None:
    allow = {str(n) for n in config.allow}
    deny = {str(n) for n in config.deny}
    if not allow and not deny:
        return
    rt = _ToolFilterRuntime(allow, deny)
    api.on(BeforeSendEvent.CHANNEL, rt.on_before_send)
