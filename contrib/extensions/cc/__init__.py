"""Claude Code compatibility package (opt-in, tier 2).

Mount this package with ``--extension contrib.extensions.cc`` to install the
Claude Code agents, commands, and plugins atoms together. The package lives in
a subdirectory so flat-file contrib auto-discovery never loads the privileged
atoms implicitly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

from . import agents, commands, plugins

class CCConfig(BaseModel):
    model_config = {"extra": "allow"}

    plugins: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    agents: dict[str, Any] = {}


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
    config_schema=CCConfig,
    tier=2,
)

MANIFESTS = (agents.MANIFEST, commands.MANIFEST, plugins.MANIFEST)


async def install(api: ExtensionAPI, config: CCConfig) -> None:
    plugin_config = plugins.PluginsConfig.model_validate(config.plugins)
    command_config = commands.CommandsConfig.model_validate(config.commands)
    agent_config = agents.AgentsConfig.model_validate(config.agents)
    plugins.install(api, plugin_config)
    await commands.install(api, command_config)
    await agents.install(api, agent_config)


__all__ = ("MANIFEST", "MANIFESTS", "install", "agents", "commands", "plugins")
