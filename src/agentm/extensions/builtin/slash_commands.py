"""Builtin slash-command input handler."""

from __future__ import annotations

import inspect
from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.events import CommandDispatchedEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="slash_commands",
    description="Rewrite escaped slashes and dispatch registered slash commands.",
    registers=("event:input", "event:command_dispatched"),
    config_schema={
        "type": "object",
        "additionalProperties": False,
    },
    api_version=1,
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    service = api.get_service("slash_commands")

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
        if not head or not isinstance(service, dict):
            return None

        commands = service.get("commands")
        if not isinstance(commands, dict):
            return None
        command = commands.get(head)
        if command is None:
            return None

        owners_by_kind = service.get("owners_by_kind")
        command_owners = (
            owners_by_kind.get("command", {})
            if isinstance(owners_by_kind, dict)
            else {}
        )
        owner = command_owners.get(head) if isinstance(command_owners, dict) else None
        apis = service.get("apis")
        owner_api = api
        if isinstance(owner, str) and isinstance(apis, dict) and owner in apis:
            owner_api = apis[owner]

        args = rest.strip()
        result = command.handler(args, owner_api)
        if inspect.isawaitable(result):
            await result
        await api.events.emit(
            CommandDispatchedEvent.CHANNEL,
            CommandDispatchedEvent(
                name=head,
                args=args,
                owner=owner if isinstance(owner, str) else "<unknown>",
            ),
        )
        messages = api.session.get_messages()
        event["handled_messages"] = messages
        return {"handled": True, "messages": messages}

    api.on("input", _on_input)
