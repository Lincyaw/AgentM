"""Slash command layer for the gateway.

See ``.claude/designs/command-routing.md`` for the architecture. The
package exposes :class:`CommandRouter` (what the gateway calls into)
plus the protocol shapes that handlers implement.
"""

from __future__ import annotations

from .protocol import (
    CommandContext,
    CommandHandler,
    CommandInbound,
    CommandInvocation,
    CommandResult,
    parse_invocation,
)
from .registry import CommandRegistry, discover_commands
from .router import CommandRouter


__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandInbound",
    "CommandInvocation",
    "CommandRegistry",
    "CommandResult",
    "CommandRouter",
    "discover_commands",
    "parse_invocation",
]
