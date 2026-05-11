"""Slash command layer for the gateway.

See ``.claude/designs/command-routing.md`` for the architecture. The
package exposes :class:`CommandRouter` (what :class:`Gateway` calls
into) plus the protocol shapes that handlers implement.
"""

from __future__ import annotations

from .protocol import (
    CommandContext,
    CommandHandler,
    CommandInvocation,
    CommandResult,
    parse_invocation,
)
from .registry import CommandRegistry, discover_commands
from .router import CommandRouter


__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandInvocation",
    "CommandRegistry",
    "CommandResult",
    "CommandRouter",
    "discover_commands",
    "parse_invocation",
]
