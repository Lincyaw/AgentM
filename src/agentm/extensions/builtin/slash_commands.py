"""Builtin slash-command input handler."""

from __future__ import annotations
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    BusPriority,
    COMMAND_PARSER,
    CommandDispatchedEvent,
    CommandDispatcher,
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
    requires=(),
    api_version=1,
    tier=1,
    provides_role=(COMMAND_PARSER,),
)


class _SlashCommandsRuntime:
    def __init__(self, session: Any) -> None:
        self._session = session
        service = session.services.get(SLASH_COMMAND_DISPATCHER_SERVICE)
        self._dispatcher = service if isinstance(service, CommandDispatcher) else None

    def install(self) -> None:
        self._session.bus.on(InputEvent.CHANNEL, self.on_input, priority=BusPriority.PRE)

    async def on_input(self, event: InputEvent) -> dict[str, Any] | None:
        text = event.text
        stripped = text.lstrip()
        if stripped.startswith("//"):
            return {"text": text.replace("//", "/", 1)}
        if not stripped.startswith("/"):
            return None

        head, _, rest = stripped[1:].partition(" ")
        if not head or self._dispatcher is None:
            return None

        args = rest.strip()
        result = await self._dispatcher.dispatch(head, args)
        if not result.handled:
            return None
        await self._session.bus.emit(
            CommandDispatchedEvent.CHANNEL,
            CommandDispatchedEvent(name=head, args=args),
        )
        return {"handled": True, "messages": result.messages}


def install(session: Any, config: SlashCommandsConfig) -> None:
    del config
    _SlashCommandsRuntime(session).install()
