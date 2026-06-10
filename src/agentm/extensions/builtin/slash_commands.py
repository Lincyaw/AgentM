"""Builtin slash-command input handler."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentm.core.abi import BusPriority
from agentm.core.abi.roles import COMMAND_PARSER, SLASH_COMMAND_DISPATCHER_SERVICE
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import CommandDispatchedEvent
from agentm.core.abi.extension import CommandDispatcher, ExtensionAPI


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


def install(api: ExtensionAPI, config: SlashCommandsConfig) -> None:
    del config
    service = api.get_service(SLASH_COMMAND_DISPATCHER_SERVICE)
    dispatcher = service if isinstance(service, CommandDispatcher) else None

    async def _on_input(event: dict[str, Any]) -> dict[str, Any] | None:
        text = event.get("text")
        if not isinstance(text, str):
            return None

        stripped = text.lstrip()
        if stripped.startswith("//"):
            event["text"] = text.replace("//", "/", 1)
            return None
        if not stripped.startswith("/"):
            return None

        head, _, rest = stripped[1:].partition(" ")
        if not head or dispatcher is None:
            return None

        args = rest.strip()
        result = await dispatcher.dispatch(head, args)
        if not result.handled:
            return None
        await api.events.emit(
            CommandDispatchedEvent.CHANNEL,
            CommandDispatchedEvent(
                name=head,
                args=args,
                owner=result.owner or "<unknown>",
            ),
        )
        messages = result.messages
        event["handled_messages"] = messages
        return {"handled": True, "messages": messages}

    api.on("input", _on_input, priority=BusPriority.PRE)
