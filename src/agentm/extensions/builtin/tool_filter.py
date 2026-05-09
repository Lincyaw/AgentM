"""Policy-gate atom for the §7 ``extensions.builtin.tool_filter`` row.

Filters the final tool catalog on ``agent_start`` after tool atoms have
registered.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import AgentStartEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_filter",
    description="Remove tools from the registered catalog by allow/deny rules.",
    registers=("event:agent_start",),
    config_schema={
        "type": "object",
        "properties": {
            "allow": {"type": "array", "items": {"type": "string"}},
            "deny": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    },
    requires=(),  # Defers filtering to agent_start so tool atoms may load in any order.
    tier=2,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    allow = {str(name) for name in config.get("allow", [])}
    deny = {str(name) for name in config.get("deny", [])}
    if not allow and not deny:
        return

    filtered = False

    def _on_agent_start(_: AgentStartEvent) -> None:
        nonlocal filtered
        if filtered:
            return
        tools = api.tools
        tool_names = {tool.name for tool in tools}
        missing_allowed = sorted(allow - tool_names)
        if missing_allowed:
            raise RuntimeError(
                "tool_filter allow-list names are not registered: "
                + ", ".join(missing_allowed)
            )
        kept = [
            tool
            for tool in tools
            if tool.name not in deny and (not allow or tool.name in allow)
        ]
        tools[:] = kept
        filtered = True

    api.on(AgentStartEvent.CHANNEL, _on_agent_start)
