"""Policy-gate atom for the §7 ``extensions.builtin.permission`` row.

Implements allow/deny filtering on the ``tool_call`` channel. ``deny`` wins
when a tool name appears in both lists.
"""

from __future__ import annotations

from typing import Any

from agentm.core.kernel import ToolCallEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="permission",
    description="Block or allow tool calls via allow/deny lists.",
    registers=("event:tool_call",),
    config_schema={
        "type": "object",
        "properties": {
            "allow": {"type": "array", "items": {"type": "string"}},
            "deny": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    },
    tier=2,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    allow = {str(name) for name in config.get("allow", [])}
    deny = {str(name) for name in config.get("deny", [])}
    if not allow and not deny:
        return

    def _on_tool_call(event: ToolCallEvent) -> dict[str, Any] | None:
        if event.tool_name in deny:
            return {
                "block": True,
                "reason": f"tool '{event.tool_name}' denied by denylist",
            }
        if allow and event.tool_name not in allow:
            return {
                "block": True,
                "reason": f"tool '{event.tool_name}' is not in allowlist",
            }
        return None

    api.on("tool_call", _on_tool_call)
