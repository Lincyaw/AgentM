"""Router — the pure dispatch decision (§3.2).

Given an inbound :class:`Envelope`, decide a small dispatch action.

Modern clients may set an explicit ``body["action"]``:

* ``submit`` — user content to be submitted as a new turn.
* ``interrupt`` — preempt the in-flight prompt.
* ``run_command`` — invoke command dispatch.
* ``interaction_response`` — resolve a pending human interaction.

For backwards compatibility, clients without explicit action still work via
legacy fields:

* ``control="interrupt"``
* ``button_value`` (approval-card click)
* slash-command content
* plain content as prompt submit

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

    SUBMIT = "submit"
    INTERACTION_RESPONSE = "interaction_response"
    INTERRUPT = "interrupt"
    RESOLVE_APPROVAL = "resolve_approval"
    RUN_COMMAND = "run_command"
    # Legacy prompt_session keeps the current behavior for compatibility with
    # non-action inbounds that still carry content/commands.
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
    if body.action is not None:
        if body.action == "submit":
            return RouterDecision(RouterAction.SUBMIT, body)
        if body.action == "run_command":
            return RouterDecision(RouterAction.RUN_COMMAND, body)
        if body.action == "interaction_response":
            return RouterDecision(RouterAction.INTERACTION_RESPONSE, body)
        if body.action == "interrupt":
            return RouterDecision(RouterAction.INTERRUPT, body)
        if body.action == "resolve_approval":
            return RouterDecision(RouterAction.RESOLVE_APPROVAL, body)
        raise ProtocolError(f"unsupported explicit action {body.action!r}")

    if body.control == "interrupt":
        return RouterDecision(RouterAction.INTERRUPT, body)
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
