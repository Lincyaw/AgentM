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
    tier=2,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    allow = {str(name) for name in config.get("allow", [])}
    deny = {str(name) for name in config.get("deny", [])}
    if not allow and not deny:
        return

    # ``api.tools`` is the live tool-catalog list (per ExtensionAPI contract);
    # ``tools[:] = kept`` mutates the registry in place so the kernel and
    # downstream extensions see the filtered set on every subsequent turn.
    tools = api.tools
    kept = [
        tool
        for tool in tools
        if tool.name not in deny and (not allow or tool.name in allow)
    ]
    tools[:] = kept
