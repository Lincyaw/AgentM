"""Builtin slash-command input handler."""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import (
    BusPriority,
    COMMAND_PARSER,
    CommandDispatchedEvent,
    CommandDispatcher,
    ExtensionAPI,
    InputEvent,
    SLASH_COMMAND_DISPATCHER_SERVICE,
)
from agentm.extensions import ExtensionManifest


class SlashCommandsConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="slash_commands",
    description="Rewrite escaped slashes and dispatch registered slash commands.",
    registers=("event:input", "event:command_dispatched"),
    config_schema=SlashCommandsConfig,
    requires=(),  # Leaf atom: dispatches commands registered by any peer.
    api_version=1,
    tier=1,
    provides_role=(COMMAND_PARSER,),
)


class _SlashCommandsRuntime:
    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api
        service = api.get_service(SLASH_COMMAND_DISPATCHER_SERVICE)
        self._dispatcher = service if isinstance(service, CommandDispatcher) else None

    def install(self) -> None:
        self._api.on(InputEvent.CHANNEL, self.on_input, priority=BusPriority.PRE)

    async def on_input(self, event: InputEvent) -> None:
        text = event.text
        stripped = text.lstrip()
        if stripped.startswith("//"):
            event.text = text.replace("//", "/", 1)
            return
        if not stripped.startswith("/"):
            return

        head, _, rest = stripped[1:].partition(" ")
        if not head or self._dispatcher is None:
            return

        args = rest.strip()
        result = await self._dispatcher.dispatch(head, args)
        if not result.handled:
            return
        await self._api.events.emit(
            CommandDispatchedEvent.CHANNEL,
            CommandDispatchedEvent(
                name=head,
                args=args,
                owner=result.owner or "<unknown>",
            ),
        )
        event.handled = True
        event.handled_messages = result.messages


def install(api: ExtensionAPI, config: SlashCommandsConfig) -> None:
    del config
    _SlashCommandsRuntime(api).install()
