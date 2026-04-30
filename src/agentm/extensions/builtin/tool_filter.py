"""Policy-gate atom for the §7 ``extensions.builtin.tool_filter`` row.

Filters the already-registered tool catalog in ``install()`` by mutating the
shared tool list in place.
"""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_filter",
    description="Remove tools from the registered catalog by allow/deny rules.",
    registers=(),
    config_schema={
        "type": "object",
        "properties": {
            "allow": {"type": "array", "items": {"type": "string"}},
            "deny": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    },
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    allow = {str(name) for name in config.get("allow", [])}
    deny = {str(name) for name in config.get("deny", [])}
    if not allow and not deny:
        return

    tools = getattr(api, "_tools", None)
    if not isinstance(tools, list):
        return

    kept = []
    for tool in tools:
        name = getattr(tool, "name", "")
        if name in deny:
            continue
        if allow and name not in allow:
            continue
        kept.append(tool)
    tools[:] = kept
