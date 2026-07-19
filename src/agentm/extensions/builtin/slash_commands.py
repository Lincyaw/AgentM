"""Builtin slash-command input handler.

TODO(migration): this is a stub. On ``main`` this atom subscribed to
``InputEvent`` (``event:input``) to rewrite escaped ``//`` slashes and dispatch
registered ``/command`` inputs through a ``CommandDispatcher`` service. This
branch has no ``InputEvent`` / ``CommandDispatchedEvent`` / ``CommandDispatcher``
/ ``SLASH_COMMAND_DISPATCHER_SERVICE`` — input arrives as typed triggers
(``push_trigger`` with a ``skip_commands`` flag), and command handling is a
runtime concern rather than a bus-event an atom can hook. There is no
equivalent subscription point to port to, so the runtime installs nothing.
Re-introduce the dispatch here once an input/command surface exists on this
branch.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority
from agentm.extensions import ExtensionManifest


class SlashCommandsConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="slash_commands",
    description="Dispatch registered slash commands (stub — no input surface yet).",
    registers=(),
    config_schema=SlashCommandsConfig,
    requires=(),
    priority=AtomInstallPriority.NORMAL,
)


def install(api: AtomAPI, config: SlashCommandsConfig) -> None:
    del api, config
    # TODO(migration): no InputEvent / command dispatcher on this branch.
    logger.debug(
        "slash_commands: no input/command surface on this branch — atom is a no-op"
    )
