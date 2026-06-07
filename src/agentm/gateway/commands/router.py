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


UNKNOWN_REPLY = (
    "Unknown command: `{raw}`\nType `/help` to see what's available."
)
"""User-visible "no such command" text. The gateway reuses this when a
slash command is unknown to *both* the gateway registry and the session's
registered set (see :meth:`_GatewayRuntime._run_command`)."""

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
        """Returns ``None`` when the gateway must not handle ``msg`` itself.

        Two distinct ``None`` cases:

        * ``msg`` is not a slash command at all (``parse_invocation``
          returns ``None``) — the gateway treats it as an ordinary prompt.
        * ``msg`` is a slash command but no *gateway* handler owns its
          name. Session-registered commands (``/compact`` and friends,
          installed by atoms like ``llm_compaction``) live inside the
          session and are dispatched there by the ``slash_commands`` floor
          atom — the gateway registry never contains those names. Returning
          ``None`` lets :meth:`_GatewayRuntime._run_command` forward the raw
          ``/...`` text to the session prompt path, whose ``input``-event
          seam then runs it. A name unknown to *both* layers reaches the
          model as text unless the gateway surfaces an "unknown command"
          diagnostic — that decision belongs to the gateway, which holds the
          per-session known-command set; the router is stateless.

        A bare ``/`` still returns a "type /help" hint (not ``None``) so it
        is handled at the gateway and never forwarded to the session."""
        inv = parse_invocation(msg)
        if inv is None:
            return None
        if not inv.name:
            return CommandResult(outbound=[ctx.reply(_EMPTY_REPLY)])
        handler = self.registry.lookup(namespace=inv.namespace, name=inv.name)
        if handler is None:
            return None
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
