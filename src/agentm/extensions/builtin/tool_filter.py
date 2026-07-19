"""Policy-gate atom for the ``extensions.builtin.tool_filter`` row.

Filters the tool set on ``before_send`` by allow/deny rules and returns
the surviving tools as a ``{"tools": [...]}`` override. Filtering here
means every LLM call sees only the permitted catalog, regardless of the
order tool-registering atoms installed in.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority, BeforeSendEvent, Tool
from agentm.extensions import ExtensionManifest


class ToolFilterConfig(BaseModel):
    model_config = {"extra": "allow"}

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
    def __init__(self, api: AtomAPI, config: ToolFilterConfig) -> None:
        self._api = api
        self._allow = {str(name) for name in config.allow}
        self._deny = {str(name) for name in config.deny}
        self._checked = False

    def active(self) -> bool:
        return bool(self._allow or self._deny)

    def install(self) -> None:
        self._api.on(BeforeSendEvent.CHANNEL, self.on_before_send)

    def on_before_send(self, event: BeforeSendEvent) -> dict[str, list[Tool]] | None:
        tools = event.tools
        if not self._checked:
            tool_names = {tool.name for tool in tools}
            missing_allowed = sorted(self._allow - tool_names)
            if missing_allowed:
                raise RuntimeError(
                    "tool_filter allow-list names are not registered: "
                    + ", ".join(missing_allowed)
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
    runtime = _ToolFilterRuntime(api, config)
    if not runtime.active():
        return
    runtime.install()
