"""Default harness slash-command dispatcher service."""

from __future__ import annotations

import inspect

from agentm.harness.extension import (
    CommandDispatchResult,
    CommandSpec,
    _ExtensionAPIImpl,
)


class HarnessCommandDispatcher:
    """Resolve registered command ownership and execute through the owner API."""

    def __init__(
        self,
        *,
        commands: dict[str, CommandSpec],
        owners_by_kind: dict[str, dict[str, str]],
        apis: dict[str, _ExtensionAPIImpl],
        fallback_owner: str,
    ) -> None:
        self._commands = commands
        self._owners_by_kind = owners_by_kind
        self._apis = apis
        self._fallback_owner = fallback_owner

    async def dispatch(self, name: str, args: str) -> CommandDispatchResult:
        command = self._commands.get(name)
        if command is None:
            return CommandDispatchResult(handled=False, owner=None, messages=[])

        owner = self._owners_by_kind.get("command", {}).get(name)
        owner_api = self._apis.get(owner or "") or self._apis.get(
            self._fallback_owner
        )
        if owner_api is None:
            return CommandDispatchResult(handled=False, owner=None, messages=[])

        result = command.handler(args, owner_api)
        if inspect.isawaitable(result):
            await result
        return CommandDispatchResult(
            handled=True,
            owner=owner,
            messages=owner_api.session.get_messages(),
        )


__all__ = ["HarnessCommandDispatcher"]
