"""CommandRouter — parses, dispatches, formats user-visible errors."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .protocol import (
    CommandContext,
    CommandInbound,
    CommandResult,
    parse_invocation,
)
from .registry import CommandRegistry


logger = logging.getLogger(__name__)


_UNKNOWN_REPLY = (
    "Unknown command: `{raw}`\nType `/help` to see what's available."
)

_EMPTY_REPLY = "Type `/help` to see available commands."


@dataclass
class CommandRouter:
    """Routes inbound messages whose content starts with ``/``.

    The router is stateless beyond the registry; per-route capabilities
    (end session, stats, extension API) are injected via
    :class:`CommandContext` by the gateway on each dispatch.
    """

    registry: CommandRegistry

    async def try_dispatch(
        self, msg: CommandInbound, ctx: CommandContext
    ) -> CommandResult | None:
        """Returns ``None`` iff ``msg`` is not a command. Otherwise
        always returns a :class:`CommandResult` — the gateway can
        rely on this to never null-check the dispatch path."""
        inv = parse_invocation(msg)
        if inv is None:
            return None
        if not inv.name:
            return CommandResult(outbound=[ctx.reply(_EMPTY_REPLY)])
        handler = self.registry.lookup(namespace=inv.namespace, name=inv.name)
        if handler is None:
            return CommandResult(
                outbound=[
                    ctx.reply(
                        _UNKNOWN_REPLY.format(raw=inv.raw),
                        kind="diagnostic_error",
                    )
                ]
            )
        try:
            return await handler.handle(inv, ctx)
        except Exception:
            logger.exception("command %r raised", inv.raw)
            return CommandResult(
                outbound=[
                    ctx.reply(
                        f"Command `{inv.raw}` failed unexpectedly. "
                        "The error has been logged.",
                        kind="diagnostic_error",
                    )
                ]
            )
