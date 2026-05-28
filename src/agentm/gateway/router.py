"""Router — the pure dispatch decision (§3.2).

Given an inbound :class:`Envelope`, decide one of three actions:

* ``RESOLVE_APPROVAL`` — the inbound carries a ``button_value`` (an
  approval-card click).
* ``RUN_COMMAND`` — the inbound's content is a slash command.
* ``PROMPT_SESSION`` — an ordinary user turn for the session.

Single function, exhaustively testable, no I/O. Only ``inbound`` kinds
reach a router in v2 — every other kind is handled by the WireServer
(``ack`` / ``ping`` / ``pong``) before it ever gets here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .wire import Envelope, InboundBody


class RouterAction(Enum):
    """What the gateway should do with an inbound."""

    RESOLVE_APPROVAL = "resolve_approval"
    RUN_COMMAND = "run_command"
    PROMPT_SESSION = "prompt_session"


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """A router verdict plus the parsed inbound body it decided on."""

    action: RouterAction
    body: InboundBody


class ProtocolError(Exception):
    """An envelope the router cannot route (wrong kind for this path)."""


def is_slash_command(content: str) -> bool:
    """True if ``content`` is a slash command (``/foo``, not ``//foo``).

    ``//`` escapes the prefix so filesystem paths typed mid-conversation
    are not hijacked as commands.
    """
    return content.startswith("/") and not content.startswith("//")


def dispatch(env: Envelope) -> RouterDecision:
    """Decide what to do with one ``inbound`` envelope (§3.2)."""
    if env.kind != "inbound":
        raise ProtocolError(f"router only handles inbound; got {env.kind!r}")
    body = InboundBody.from_dict(env.body if isinstance(env.body, dict) else {})
    if body.button_value:
        return RouterDecision(RouterAction.RESOLVE_APPROVAL, body)
    if is_slash_command(body.content):
        return RouterDecision(RouterAction.RUN_COMMAND, body)
    return RouterDecision(RouterAction.PROMPT_SESSION, body)


__all__ = [
    "ProtocolError",
    "RouterAction",
    "RouterDecision",
    "dispatch",
    "is_slash_command",
]
