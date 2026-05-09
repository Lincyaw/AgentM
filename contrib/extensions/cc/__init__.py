"""Claude Code compatibility package (opt-in, tier 2).

Mount this package with ``--extension contrib.extensions.cc`` to install the
Claude Code agents, commands, and plugins atoms together. The package lives in
a subdirectory so flat-file contrib auto-discovery never loads the privileged
atoms implicitly.
"""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

from . import agents, commands, plugins

MANIFEST = ExtensionManifest(
    name="cc",
    description="Install Claude Code compatibility atoms as one package.",
    registers=tuple(
        dict.fromkeys(
            [
                *plugins.MANIFEST.registers,
                *commands.MANIFEST.registers,
                *agents.MANIFEST.registers,
            ]
        )
    ),
    config_schema={"type": "object", "additionalProperties": True},
    tier=2,
)

MANIFESTS = (agents.MANIFEST, commands.MANIFEST, plugins.MANIFEST)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    plugin_config = dict(config.get("plugins", {}))
    command_config = dict(config.get("commands", {}))
    agent_config = dict(config.get("agents", {}))
    plugins.install(api, plugin_config)
    await commands.install(api, command_config)
    await agents.install(api, agent_config)


__all__ = ("MANIFEST", "MANIFESTS", "install", "agents", "commands", "plugins")
