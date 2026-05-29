"""Command protocol — one shape, four handler kinds.

A slash command is any inbound message whose content starts with
``/`` (and not ``//`` — the second character escapes the prefix so
filesystem paths typed mid-conversation don't get hijacked).

Two dispatch modes:

* ``control`` — handler mutates gateway/session state, optionally
  emits outbound messages, and the loop returns *without* calling
  :meth:`AgentSession.prompt`. The user-typed command text never
  reaches the LLM.
* ``prompt`` — handler returns ``expanded_prompt`` (the template
  applied to the user's args); the gateway rewrites the inbound
  message and falls through to the normal session-prompt path. The
  *expanded* text reaches the LLM; the original ``/foo …`` does not.

Unknown commands are rejected with a user-visible reply. This is
deliberate — silently forwarding ``/foo`` to the model when the user
clearly intended a command is the kind of failure mode that exposes
private prompts. The router prefers a polite "no such command" over
"the LLM has no idea what you meant either."

The :class:`CommandContext` facade is intentionally narrow. Handlers
should never receive a gateway object directly; they get the few
capabilities they need (end the session, read stats, list commands,
reach the live ExtensionAPI). Mirrors the §11 ``ExtensionAPI`` pattern:
atoms reach gateway internals only through documented services.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ..wire.types import OutboundBody


CommandKind = Literal["control", "prompt"]


async def _default_switch_model(_name: str) -> tuple[bool, str]:
    """Default ``switch_model`` capability — model switching is unsupported on
    this context (e.g. a unit-test stub). The gateway injects a real one."""
    return (False, "model switching is not supported in this context")


@dataclass(frozen=True, slots=True)
class CommandInbound:
    """The slash-command-relevant slice of an inbound envelope.

    Built by the gateway from the inbound :class:`Envelope` +
    :class:`InboundBody`; handlers read these fields rather than the
    full wire envelope.
    """

    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    content: str
    thread_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    """Parsed user invocation. Created by :func:`parse_invocation`."""

    raw: str
    """The original message content (``/foo bar``)."""

    name: str
    """The command name, lowercased. ``""`` if the message was just ``/``."""

    namespace: str | None
    """Namespace from ``/<ns>:<name>`` syntax; ``None`` for builtins."""

    args: str
    """Everything after the first whitespace, preserved verbatim. Empty
    when the command has no args. *Not* shell-split — handlers parse
    their own args."""

    inbound: CommandInbound
    """The originating inbound — handlers read ``sender_id`` / ``chat_id``
    when needed."""


@dataclass(slots=True)
class CommandResult:
    """What a handler returns.

    For ``control`` handlers, set ``outbound`` (messages to publish)
    and optionally ``side_effect`` (an async callback the router runs
    after publishing). ``expanded_prompt`` must be ``None``.

    For ``prompt`` handlers, set ``expanded_prompt`` to the rewritten
    text the gateway should feed to ``session.prompt(...)``. Other
    fields are ignored.
    """

    outbound: list[OutboundBody] = field(default_factory=list)
    side_effect: Callable[["Any"], Awaitable[None]] | None = None
    expanded_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Narrow facade passed to every handler — see module docstring."""

    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    thread_id: str | None
    end_session: Callable[[], Awaitable[None]]
    """Tear down the active ``AgentSession`` for this chat. ``/new`` keeps
    the :class:`ChatSessionMap` entry (next message resumes from
    transcript); ``/end`` additionally clears it (next message starts
    fresh). The flavour is selected via ``forget_chat_mapping`` below."""

    forget_chat_mapping: Callable[[], Awaitable[None]]
    """Clear the persistent ``ChatSessionMap`` entry for this chat. Used
    by ``/end`` after ``end_session`` so the next message starts cold."""

    get_route_stats: Callable[[], dict[str, Any]]
    """Returns a snapshot dict (``session_id``, ``turn_count``,
    ``pending_approvals``) for ``/status``-style introspection."""

    list_commands: Callable[[], list["CommandHandler"]]
    """Returns every handler the router currently knows about. Used by
    ``/help``. The list is a snapshot — handlers must not mutate it."""

    get_extension_api: Callable[[], Any | None] = lambda: None
    """Live ``ExtensionAPI`` for this chat's session, or ``None`` if the
    session has not been created yet. Used by ``/atom:*`` commands."""

    list_models: Callable[[], tuple[str, list[str]]] = lambda: ("", [])
    """Returns ``(active_model_name, available_profile_names)`` for ``/model``.
    Names are the ``[models.<name>]`` keys from ``config.toml``."""

    switch_model: Callable[[str], Awaitable[tuple[bool, str]]] = (
        lambda _name: _default_switch_model(_name)
    )
    """Switch the active model profile and restart this chat's session
    (keeps transcript). Returns ``(ok, message)`` — ``message`` is the
    resolved model name on success, or an error reason on failure."""

    def reply(self, text: str, **meta: Any) -> OutboundBody:
        """Build a plain ``assistant_text`` outbound back to this chat."""
        metadata: dict[str, Any] = {"kind": "assistant_text"}
        metadata.update(meta)
        return OutboundBody(
            channel=self.channel,
            chat_id=self.chat_id,
            content=text,
            thread_id=self.thread_id,
            metadata=metadata,
        )


@runtime_checkable
class CommandHandler(Protocol):
    """The contract every command implements.

    ``name`` is lowercase, no leading slash. ``namespace`` is ``None``
    for builtin/control commands and the literal source name otherwise
    (``"skill"``, ``"atom"``, plugin identifier).
    """

    name: str
    namespace: str | None
    summary: str
    kind: CommandKind

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult: ...


# --- parsing ----------------------------------------------------------


def parse_invocation(msg: CommandInbound) -> CommandInvocation | None:
    """Return a :class:`CommandInvocation` if ``msg`` looks like a
    command, otherwise ``None``.

    Rules:

    * Must start with exactly one ``/``. ``//foo`` (escaped filesystem
      path) returns ``None``.
    * Bare ``/`` (just the slash, no name) returns an invocation with
      empty ``name`` — the router uses that to surface a "type
      ``/help``" hint.
    * Names are lowercased; args are preserved verbatim.
    * ``/<ns>:<name>`` splits on the first ``:`` only — names with
      colons in them (rare, but possible) survive after the first.
    """
    content = msg.content
    if not content.startswith("/"):
        return None
    if content.startswith("//"):
        return None
    stripped = content[1:]
    head, _, args = stripped.partition(" ")
    args = args.lstrip()
    namespace: str | None
    name: str
    if ":" in head:
        ns_part, _, name_part = head.partition(":")
        namespace = ns_part.lower() or None
        name = name_part.lower()
    else:
        namespace = None
        name = head.lower()
    return CommandInvocation(
        raw=content,
        name=name,
        namespace=namespace,
        args=args,
        inbound=msg,
    )


__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandInbound",
    "CommandInvocation",
    "CommandKind",
    "CommandResult",
    "parse_invocation",
]
