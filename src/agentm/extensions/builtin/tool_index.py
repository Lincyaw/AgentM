"""Builtin ``tool_index`` atom: inject registered tools into the system prompt.

Appends an ``<available_tools>`` block listing every registered tool's
name and description to the system prompt at ``BeforeAgentStartEvent``
time — after all atoms have finished ``install`` and registered their
tools.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from pydantic import BaseModel

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest


class ToolIndexConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="tool_index",
    description="Append an <available_tools> index to the system prompt.",
    registers=("event:before_agent_start",),
    config_schema=ToolIndexConfig,
    requires=(),
)


class _ToolIndexRuntime:
    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api

    def install(self) -> None:
        self._api.on(BeforeAgentStartEvent.CHANNEL, self.inject)

    def inject(self, event: BeforeAgentStartEvent) -> dict[str, str] | None:
        tools = self._api.tools
        if not tools:
            return None
        lines = [
            "# Tools",
            "",
            "You have the following tools available. Use the appropriate tool "
            "when the task calls for it — prefer tools over generating answers "
            "from memory when a tool can provide authoritative, up-to-date "
            "information. You may call multiple tools in a single turn if "
            "needed.",
            "",
            "<available_tools>",
        ]
        for tool in tools:
            lines.append("  <tool>")
            lines.append(f"    <name>{escape(tool.name)}</name>")
            lines.append(f"    <description>{escape(tool.description)}</description>")
            lines.append("  </tool>")
        lines.append("</available_tools>")
        block = "\n".join(lines)
        current = event.system or ""
        updated = f"{current}\n\n{block}" if current else block
        event.system = updated
        return {"system": updated}


def install(api: ExtensionAPI, _config: ToolIndexConfig) -> None:
    _ToolIndexRuntime(api).install()
