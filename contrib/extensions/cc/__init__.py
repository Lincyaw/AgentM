"""Claude Code compatibility package (opt-in, tier 2).

Mount this package with ``--extension contrib.extensions.cc`` to install the
Claude Code agents, commands, and plugins atoms together. The package lives in
a subdirectory so flat-file contrib auto-discovery never loads the privileged
atoms implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from agentm.extensions import ExtensionManifest
from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI

from . import agents, commands, plugins

class CCConfig(BaseModel):
    model_config = {"extra": "allow"}

    plugins: dict[str, Any] = Field(default_factory=dict)
    commands: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, Any] = Field(default_factory=dict)
    inherit_startup_claude: bool = True


_MAX_STARTUP_CLAUDE_BYTES = 262_144


def _load_startup_claude(cwd: str) -> str:
    """Return startup-directory ``CLAUDE.md`` content, if present."""

    path = Path(cwd) / "CLAUDE.md"
    try:
        if not path.is_file():
            return ""
        if path.stat().st_size > _MAX_STARTUP_CLAUDE_BYTES:
            logger.warning(
                "cc: skipping startup CLAUDE.md larger than {} bytes: {}",
                _MAX_STARTUP_CLAUDE_BYTES,
                path,
            )
            return ""
        return path.read_text(encoding="utf-8").rstrip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("cc: failed to read startup CLAUDE.md at {}: {}", path, exc)
        return ""


def _format_startup_claude_prompt(body: str) -> str:
    return "\n\n".join(("Startup directory CLAUDE.md:", body))

MANIFEST = ExtensionManifest(
    name="cc",
    description="Install Claude Code compatibility atoms as one package.",
    registers=tuple(
        dict.fromkeys(
            [
                *plugins.MANIFEST.registers,
                *commands.MANIFEST.registers,
                *agents.MANIFEST.registers,
                f"event:{BeforeAgentStartEvent.CHANNEL}",
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

    startup_claude_body = (
        _load_startup_claude(api.cwd) if config.inherit_startup_claude else ""
    )
    startup_claude = (
        _format_startup_claude_prompt(startup_claude_body)
        if startup_claude_body
        else ""
    )
    if startup_claude:

        def _inject_startup_claude(
            event: BeforeAgentStartEvent,
        ) -> None:
            current = str(event.system or "")
            updated = f"{startup_claude}\n\n{current}" if current else startup_claude
            event.system = updated

        api.on(BeforeAgentStartEvent.CHANNEL, _inject_startup_claude)

__all__ = ("MANIFEST", "MANIFESTS", "install", "agents", "commands", "plugins")
